import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import networkx as nx
from datetime import datetime
import logging
# Setup debug logging
logging.basicConfig(filename='debug_log.txt', level=logging.DEBUG, format='%(asctime)s %(message)s')

# At the top of the file, add a patch to suppress tqdm and unwanted stdout/stderr output
import sys
import contextlib
import os

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

class ReasoningRulesLoader:
    """Loads and manages reasoning rules from JSON file"""
    
    def __init__(self, rules_file_path: str = "reasoning_rules.json"):
        self.rules_file_path = rules_file_path
        self.reasoning_rules = {}
        self.rule_costs = {}
        self.load_rules()
    
    def load_rules(self):
        """Load reasoning rules from JSON file"""
        try:
            with open(self.rules_file_path, 'r', encoding='utf-8') as f:
                rules_data = json.load(f)
            
            # Flatten the nested structure and extract costs
            for category, rules in rules_data.items():
                for rule_id, rule_info in rules.items():
                    self.reasoning_rules[rule_id] = rule_info
                    self.rule_costs[rule_id] = rule_info.get('cost', 2)  # Default cost 2
            
            print(f"✅ Loaded {len(self.reasoning_rules)} reasoning rules from {self.rules_file_path}")
            
        except FileNotFoundError:
            print(f"❌ Rules file not found: {self.rules_file_path}")  # Retry loading
            
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON file: {e}")
            raise
    
    def create_default_rules_file(self):
        """Create a default rules file if none exists"""
        default_rules = {
            "foundational_logical": {
                "causation_analysis": {
                    "rule": "Correlation does not imply causation - systematically examine confounding variables, reverse causality, temporal sequences, and mechanism plausibility",
                    "framework": "Ask: What's the temporal order? What's the proposed mechanism? What else could explain this correlation? Could the supposed effect actually be the cause?",
                    "keywords": ["cause", "causation", "correlation", "because", "due to", "leads to", "mechanism", "temporal"],
                    "complexity": "medium",
                    "cost": 2
                },
                "evidence_hierarchies": {
                    "rule": "Not all evidence is equal - understand evidence hierarchies, methodological quality, replication status, and convergent validity across multiple sources",
                    "framework": "Ask: What type of evidence is this? How was it gathered? Has it been replicated? Do multiple independent sources converge?",
                    "keywords": ["evidence", "proof", "study", "research", "methodology", "replication", "meta-analysis"],
                    "complexity": "low",
                    "cost": 2
                }
            }
        }
        
        with open(self.rules_file_path, 'w', encoding='utf-8') as f:
            json.dump(default_rules, f, indent=2)
        
        print(f"Created default rules file: {self.rules_file_path}")
    
    def get_all_rules(self) -> Dict:
        """Get all reasoning rules"""
        return self.reasoning_rules
    
    def get_rule_costs(self) -> Dict:
        """Get all rule costs"""
        return self.rule_costs
    
    def get_rules_by_complexity(self, complexity_level: str) -> List[str]:
        """Get rule IDs by complexity level"""
        return [rule_id for rule_id, rule_data in self.reasoning_rules.items() 
                if rule_data.get('complexity') == complexity_level]
    
    def reload_rules(self):
        """Reload rules from file (useful for hot-reloading)"""
        self.reasoning_rules = {}
        self.rule_costs = {}
        self.load_rules()

class DebateMemoryGraph:
    """Graph-based memory system for storing and retrieving debate history"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.turn_counter = 0
        self.argument_embedder = SentenceTransformer('all-mpnet-base-v2')
        self.last_argument = {}  # Cache last argument per side
        self.last_embedding = {}  # Cache last embedding per side
        self.frame_usage = {}  # Track frame usage and effectiveness
    
    def clear_history(self):
        """Clear all existing debate history"""
        self.graph.clear()
        self.turn_counter = 0
        print("🧠 Memory graph cleared - fresh start")
    
    def add_argument(self, side: str, argument: str, rules_used: List[str], 
                    quality_score: float, rl_result: Dict, stamina_before: int, 
                    stamina_after: int, synthesis_level: str, frames_used: List[str] = None) -> str:
        """Add an argument to the memory graph"""
        
        # Prevent argument echoing: do not add if identical or highly similar to last argument from same side
        side_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]['side'] == side]
        if side_nodes:
            last_node = max(side_nodes, key=lambda n: self.graph.nodes[n]['turn'])
            last_arg = self.graph.nodes[last_node]['argument']
            with suppress_stdout_stderr():
                last_emb = np.array(self.graph.nodes[last_node]['embedding'])
                new_emb = self.argument_embedder.encode([argument], show_progress_bar=False)[0]
            similarity = np.dot(last_emb, new_emb) / (np.linalg.norm(last_emb) * np.linalg.norm(new_emb))
            if argument.strip() == last_arg.strip() or similarity > 0.97:
                logging.debug(f"Echo/prevented: {argument}")
                return last_node
        # Cache last argument/embedding
        self.last_argument[side] = argument
        with suppress_stdout_stderr():
            self.last_embedding[side] = self.argument_embedder.encode([argument], show_progress_bar=False)[0]
        
        self.turn_counter += 1
        node_id = f"{side}_turn_{self.turn_counter}"
        
        # Create argument embedding for similarity search
        with suppress_stdout_stderr():
            embedding = self.argument_embedder.encode([argument], show_progress_bar=False)[0]
        
        # Add node with comprehensive data
        self.graph.add_node(node_id, 
                           side=side,
                           turn=self.turn_counter,
                           argument=argument,
                           rules_used=rules_used,
                           quality_score=quality_score,
                           rl_change=rl_result.get('reward_change', 0),
                           rl_category=rl_result.get('category', 'unknown'),
                           stamina_before=stamina_before,
                           stamina_after=stamina_after,
                           synthesis_level=synthesis_level,
                           timestamp=datetime.now().isoformat(),
                           embedding=embedding.tolist(),
                           word_count=len(argument.split()),
                           frames_used=frames_used or [])
        
        # Add edges to previous arguments (temporal flow)
        previous_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]['turn'] == self.turn_counter - 1]
        for prev_node in previous_nodes:
            self.graph.add_edge(prev_node, node_id, relationship="temporal_sequence")
        
        # Add response edges (if this is a response to opponent)
        if self.turn_counter > 1:
            opponent_side = "con" if side == "pro" else "pro"
            opponent_nodes = [n for n in self.graph.nodes() 
                            if self.graph.nodes[n]['side'] == opponent_side and 
                            self.graph.nodes[n]['turn'] == self.turn_counter - 1]
            for opp_node in opponent_nodes:
                self.graph.add_edge(opp_node, node_id, relationship="response_to")
        
        # Track frame usage and effectiveness
        if frames_used:
            for frame in frames_used:
                if frame not in self.frame_usage:
                    self.frame_usage[frame] = {'count': 0, 'total_quality': 0.0}
                self.frame_usage[frame]['count'] += 1
                self.frame_usage[frame]['total_quality'] += quality_score
        
        return node_id
    
    def get_side_history(self, side: str, max_turns: int = 10) -> List[Dict]:
        """Get recent history for a specific side"""
        
        side_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]['side'] == side]
        side_nodes.sort(key=lambda x: self.graph.nodes[x]['turn'], reverse=True)
        
        history = []
        for node in side_nodes[:max_turns]:
            node_data = self.graph.nodes[node]
            history.append({
                'turn': node_data['turn'],
                'argument': node_data['argument'],
                'rules_used': node_data['rules_used'],
                'quality_score': node_data['quality_score'],
                'rl_category': node_data['rl_category'],
                'rl_change': node_data['rl_change'],
                'stamina_before': node_data['stamina_before'],
                'stamina_after': node_data['stamina_after']
            })
        
        return history
    
    def get_opponent_history(self, current_side: str, max_turns: int = 10) -> List[Dict]:
        """Get recent history of the opponent"""
        
        opponent_side = "con" if current_side == "pro" else "pro"
        return self.get_side_history(opponent_side, max_turns)
    
    def get_conversation_flow(self, max_turns: int = 15) -> List[Dict]:
        """Get chronological conversation flow"""
        
        all_nodes = list(self.graph.nodes())
        all_nodes.sort(key=lambda x: self.graph.nodes[x]['turn'])
        
        flow = []
        for node in all_nodes[-max_turns:]:
            node_data = self.graph.nodes[node]
            flow.append({
                'turn': node_data['turn'],
                'side': node_data['side'],
                'argument': node_data['argument'],
                'quality_score': node_data['quality_score'],
                'rl_category': node_data['rl_category'],
                'synthesis_level': node_data['synthesis_level']
            })
        
        return flow
    
    def find_similar_arguments(self, current_argument: str, same_side: bool = True, 
                              similarity_threshold: float = 0.7) -> List[Dict]:
        """Find similar arguments using embedding similarity"""
        
        if not self.graph.nodes():
            return []
        
        with suppress_stdout_stderr():
            current_embedding = self.argument_embedder.encode([current_argument], show_progress_bar=False)[0]
        
        similar_args = []
        
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            stored_embedding = np.array(node_data['embedding'])
            
            # Calculate cosine similarity
            similarity = np.dot(current_embedding, stored_embedding) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(stored_embedding)
            )
            
            if similarity >= similarity_threshold:
                similar_args.append({
                    'turn': node_data['turn'],
                    'side': node_data['side'],
                    'argument': node_data['argument'],
                    'similarity': float(similarity),
                    'quality_score': node_data['quality_score'],
                    'rl_category': node_data['rl_category']
                })
        
        # Sort by similarity
        similar_args.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_args[:5]  # Return top 5 similar arguments
    
    def get_quality_trends(self, side: str) -> Dict:
        """Analyze quality trends for a side"""
        
        side_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]['side'] == side]
        side_nodes.sort(key=lambda x: self.graph.nodes[x]['turn'])
        
        if len(side_nodes) < 2:
            return {"trend": "insufficient_data", "recent_quality": 0, "quality_history": []}
        
        quality_scores = [self.graph.nodes[n]['quality_score'] for n in side_nodes]
        recent_scores = quality_scores[-3:] if len(quality_scores) >= 3 else quality_scores
        earlier_scores = quality_scores[:-3] if len(quality_scores) >= 6 else quality_scores[:-1]
        
        recent_avg = sum(recent_scores) / len(recent_scores)
        earlier_avg = sum(earlier_scores) / len(earlier_scores) if earlier_scores else recent_avg
        
        if recent_avg > earlier_avg + 0.5:
            trend = "improving"
        elif recent_avg < earlier_avg - 0.5:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "recent_quality": recent_avg,
            "earlier_quality": earlier_avg,
            "quality_history": quality_scores,
            "total_arguments": len(side_nodes)
        }
    
    def get_rule_effectiveness(self, side: str) -> Dict:
        """Analyze which reasoning rules have been most effective"""
        
        side_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]['side'] == side]
        
        rule_performance = defaultdict(list)
        
        for node in side_nodes:
            node_data = self.graph.nodes[node]
            rules = node_data['rules_used']
            quality = node_data['quality_score']
            
            for rule in rules:
                rule_performance[rule].append(quality)
        
        rule_stats = {}
        for rule, scores in rule_performance.items():
            rule_stats[rule] = {
                "avg_quality": sum(scores) / len(scores),
                "usage_count": len(scores),
                "best_score": max(scores),
                "worst_score": min(scores)
            }
        
        # Sort by average quality
        sorted_rules = sorted(rule_stats.items(), key=lambda x: x[1]['avg_quality'], reverse=True)
        
        return {
            "best_rules": sorted_rules[:3],
            "worst_rules": sorted_rules[-3:] if len(sorted_rules) > 3 else [],
            "total_rules_used": len(rule_stats)
        }
    
    def get_memory_summary(self, side: str) -> str:
        """Generate a concise memory summary for the current side"""
        
        if not self.graph.nodes():
            return "No previous arguments in memory."
        
        side_history = self.get_side_history(side, 5)
        opponent_history = self.get_opponent_history(side, 5)
        quality_trends = self.get_quality_trends(side)
        rule_effectiveness = self.get_rule_effectiveness(side)
        
        summary_parts = []
        
        # Own performance
        if side_history:
            last_quality = side_history[0]['quality_score']
            summary_parts.append(f"Your last argument scored {last_quality:.1f}/10 ({side_history[0]['rl_category']})")
            
            if quality_trends['trend'] == 'improving':
                summary_parts.append("Your quality is improving")
            elif quality_trends['trend'] == 'declining':
                summary_parts.append("Your quality is declining - adjust strategy")
        
        # Best performing rules
        if rule_effectiveness['best_rules']:
            best_rule = rule_effectiveness['best_rules'][0]
            summary_parts.append(f"Your most effective rule: {best_rule[0]} (avg {best_rule[1]['avg_quality']:.1f})")
        
        # Recent opponent moves
        if opponent_history:
            opp_last = opponent_history[0]
            summary_parts.append(f"Opponent's last: {opp_last['quality_score']:.1f}/10 ({opp_last['rl_category']})")
        
        # Flow insight
        flow = self.get_conversation_flow(10)
        if len(flow) >= 4:
            recent_quality_avg = sum(turn['quality_score'] for turn in flow[-4:]) / 4
            summary_parts.append(f"Recent debate quality: {recent_quality_avg:.1f}/10")
        
        return " | ".join(summary_parts)
    
    def get_memory_insights(self, side: str) -> dict:
        """Return structured memory insights for the debater"""
        insights = {}
        # Top arguments by quality (own and opponent)
        side_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]['side'] == side]
        opp_side = 'con' if side == 'pro' else 'pro'
        opp_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]['side'] == opp_side]
        
        top_own = sorted(side_nodes, key=lambda n: self.graph.nodes[n]['quality_score'], reverse=True)[:3]
        top_opp = sorted(opp_nodes, key=lambda n: self.graph.nodes[n]['quality_score'], reverse=True)[:3]
        insights['top_own_arguments'] = [self.graph.nodes[n]['argument'] for n in top_own]
        insights['top_opp_arguments'] = [self.graph.nodes[n]['argument'] for n in top_opp]
        
        # Most/least effective rules
        rule_eff = self.get_rule_effectiveness(side)
        insights['best_rules'] = rule_eff['best_rules']
        insights['worst_rules'] = rule_eff['worst_rules']
        
        # Repetition: arguments with high similarity
        repeated = []
        for n in side_nodes:
            arg = self.graph.nodes[n]['argument']
            similars = self.find_similar_arguments(arg, same_side=True, similarity_threshold=0.92)
            if len(similars) > 1:
                repeated.append(arg)
        insights['repeated_arguments'] = repeated[:3]
        
        # Quality trend
        insights['quality_trend'] = self.get_quality_trends(side)
        
        # Opponent's best moves
        insights['opp_best_moves'] = [self.graph.nodes[n]['argument'] for n in top_opp]
        return insights
    
    def batch_encode_arguments(self, arguments: List[str]):
        """Efficient batch embedding for a list of arguments"""
        with suppress_stdout_stderr():
            return self.argument_embedder.encode(arguments, show_progress_bar=False)
    
    def update_from_reflection(self, side: str, reflection: str):
        # Example: If reflection says 'repetitive', mark last argument as less effective
        if not reflection:
            return
        rc = reflection.lower()
        if 'repetitive' in rc or 'repetition' in rc or 'same' in rc:
            side_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]['side'] == side]
            if side_nodes:
                last_node = max(side_nodes, key=lambda n: self.graph.nodes[n]['turn'])
                self.graph.nodes[last_node]['quality_score'] = max(0, self.graph.nodes[last_node]['quality_score'] - 1)
        # If reflection says 'new theme', add a tag
        if 'theme' in rc or 'new theme' in rc:
            side_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]['side'] == side]
            if side_nodes:
                last_node = max(side_nodes, key=lambda n: self.graph.nodes[n]['turn'])
                self.graph.nodes[last_node]['theme'] = 'new'
    
    def get_frame_analytics(self):
        """Return analytics on frame usage and effectiveness"""
        analytics = []
        for frame, data in self.frame_usage.items():
            avg_quality = data['total_quality'] / data['count'] if data['count'] else 0
            analytics.append({'frame': frame, 'count': data['count'], 'avg_quality': avg_quality})
        analytics.sort(key=lambda x: x['avg_quality'], reverse=True)
        return analytics

class StrictRewardPunishmentSystem:
    """Strict RL system with heavy penalties for poor reasoning, now with memory-guided bonuses/penalties (no model learning)"""
    
    def __init__(self):
        # Reward/punishment configuration - STRICT PENALTIES
        self.quality_thresholds = {
            "exceptional": {"min_score": 8.5, "reward": 6},      # Rare, high reward
            "good": {"min_score": 7.5, "reward": 3},            # Above average
            "acceptable": {"min_score": 6.5, "reward": 0}, 
            "bad":{"min_score":5.5,"reward":-3},
            "poor": {"min_score": 3.0, "punishment": -8},       # Heavy penalty
            "terrible": {"min_score": 0.0, "punishment": -15}   # Severe penalty
        }
        
        # Exchange rates - more generous rewards, strict punishments
        self.stamina_exchange_rate = 3  # 3 reward points = 9 stamina (3x multiplier for meaningful recovery)
        self.punishment_exchange_rate = 1  # 1 punishment point = 1 stamina loss
        
        # Tracking
        self.pro_reward_bank = 0
        self.con_reward_bank = 0
        self.pro_punishment_debt = 0
        self.con_punishment_debt = 0
        
        # History for analysis
        self.evaluation_history = {"pro": [], "con": []}
        
    def calculate_quality_score(self, argument: str, rules_used: List[Dict], 
                              mentor_evaluation: Dict) -> float:
        """Calculate argument quality score based on multiple factors"""
        
        # Base score from mentor evaluation (0-10)
        mentor_score = mentor_evaluation.get('reasoning_quality', 5.0)
        
        # Rule complexity bonus (higher complexity = higher potential score)
        complexity_weights = {
            "low": 0.5,
            "medium": 1.0, 
            "high": 2.0,
            "very_high": 3.0
        }
        
        complexity_bonus = 0
        for rule in rules_used:
            complexity = rule.get('complexity', 'medium')
            complexity_bonus += complexity_weights.get(complexity, 1.0)
        
        # Normalize complexity bonus (max 3 points)
        complexity_bonus = min(complexity_bonus / len(rules_used) if rules_used else 0, 3.0)
        
        # Synthesis and conclusion bonus
        synthesis_bonus = mentor_evaluation.get('synthesis_quality', 0) * 0.5
        
        # Final score (0-10 scale)
        final_score = min(10.0, mentor_score + complexity_bonus + synthesis_bonus)
        
        return final_score
    
    def evaluate_and_reward(self, side: str, argument: str, rules_used: List[Dict], 
                           mentor_evaluation: Dict, memory_insights: dict = None) -> Dict:
        """Evaluate argument and apply rewards/punishments, using memory insights for better guidance"""
        quality_score = self.calculate_quality_score(argument, rules_used, mentor_evaluation)
        base_score = quality_score
        bonus = 0
        penalty = 0
        feedback = mentor_evaluation.get('feedback', '').lower()
        # --- Memory-guided bonuses/penalties ---
        if memory_insights:
            # 1. Penalize repetition
            if argument in memory_insights.get('repeated_arguments', []):
                penalty -= 1.5  # discourage repetition
            # 2. Reward novelty (not in repeated)
            elif len(argument.split()) > 5:
                bonus += 0.5  # encourage new, substantive arguments
            # 3. Reward use of best rules
            best_rules = [r[0] for r in memory_insights.get('best_rules', [])]
            used_rule_ids = [r.get('rule_id', r) for r in rules_used]
            if any(rule in used_rule_ids for rule in best_rules):
                bonus += 0.5
            # 4. Penalize overuse of worst rules
            worst_rules = [r[0] for r in memory_insights.get('worst_rules', [])]
            if any(rule in used_rule_ids for rule in worst_rules):
                penalty -= 0.5
            # 5. Reward improvement over recent trend
            trend = memory_insights.get('quality_trend', {})
            recent_avg = trend.get('recent_quality', 0)
            if quality_score > recent_avg + 0.5:
                bonus += 0.5
            elif quality_score < recent_avg - 0.5:
                penalty -= 0.5
        # --- Mentor feedback cues ---
        if any(word in feedback for word in ['novel', 'adapt', 'integrat']):
            bonus += 0.5
        if any(word in feedback for word in ['repeat', 'redundant', 'ignore', 'weak']):
            penalty -= 0.5
        # Final score with bonuses/penalties
        final_score = min(10.0, max(0.0, base_score + bonus + penalty))
        # Determine reward/punishment category
        category = "terrible"
        reward_change = -15  # Default severe punishment
        for cat, config in self.quality_thresholds.items():
            if final_score >= config["min_score"]:
                category = cat
                if "reward" in config:
                    reward_change = config["reward"]
                else:
                    reward_change = config["punishment"]
                break
        # Apply rewards/punishments
        if reward_change > 0:
            if side == "pro":
                self.pro_reward_bank += reward_change
            else:
                self.con_reward_bank += reward_change
        else:
            if side == "pro":
                self.pro_punishment_debt += abs(reward_change)
            else:
                self.con_punishment_debt += abs(reward_change)
        # Record evaluation
        evaluation_record = {
            "quality_score": final_score,
            "category": category,
            "reward_change": reward_change,
            "rules_used": [rule.get('rule_id', 'unknown') for rule in rules_used],
            "mentor_scores": mentor_evaluation,
            "argument_length": len(argument.split()),
            "bonus": bonus,
            "penalty": penalty,
            "memory_guided": True if memory_insights else False,
            "timestamp": time.time()
        }
        self.evaluation_history[side].append(evaluation_record)
        return {
            "quality_score": final_score,
            "category": category,
            "reward_change": reward_change,
            "can_exchange": self.can_exchange_for_stamina(side),
            "punishment_debt": self.get_punishment_debt(side),
            "reward_bank": self.get_reward_bank(side)
        }
    
    def can_exchange_for_stamina(self, side: str) -> bool:
        """Check if side can exchange rewards for stamina"""
        reward_bank = self.pro_reward_bank if side == "pro" else self.con_reward_bank
        return reward_bank >= self.stamina_exchange_rate
    
    def exchange_rewards_for_stamina(self, side: str) -> int:
        """Exchange reward points for stamina with 3x multiplier"""
        reward_bank = self.pro_reward_bank if side == "pro" else self.con_reward_bank
        
        if reward_bank >= self.stamina_exchange_rate:
            stamina_exchanges = reward_bank // self.stamina_exchange_rate
            rewards_spent = stamina_exchanges * self.stamina_exchange_rate
            stamina_gained = stamina_exchanges * 3  # 3x multiplier: 3 rewards = 9 stamina
            
            if side == "pro":
                self.pro_reward_bank -= rewards_spent
            else:
                self.con_reward_bank -= rewards_spent
                
            return stamina_gained
        return 0
    
    def apply_punishment_debt(self, side: str) -> int:
        """Apply punishment debt as stamina loss"""
        punishment_debt = self.pro_punishment_debt if side == "pro" else self.con_punishment_debt
        
        if punishment_debt >= self.punishment_exchange_rate:
            stamina_lost = punishment_debt // self.punishment_exchange_rate
            punishment_applied = stamina_lost * self.punishment_exchange_rate
            
            if side == "pro":
                self.pro_punishment_debt -= punishment_applied
            else:
                self.con_punishment_debt -= punishment_applied
                
            return stamina_lost
        return 0
    
    def get_reward_bank(self, side: str) -> int:
        """Get current reward bank"""
        return self.pro_reward_bank if side == "pro" else self.con_reward_bank
    
    def get_punishment_debt(self, side: str) -> int:
        """Get current punishment debt"""
        return self.pro_punishment_debt if side == "pro" else self.con_punishment_debt
    
    def get_net_performance(self, side: str) -> float:
        """Get net performance score (rewards - punishments)"""
        rewards = self.pro_reward_bank if side == "pro" else self.con_reward_bank
        punishments = self.pro_punishment_debt if side == "pro" else self.con_punishment_debt
        return rewards - punishments
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        return {
            "pro": {
                "reward_bank": self.pro_reward_bank,
                "punishment_debt": self.pro_punishment_debt,
                "net_score": self.get_net_performance("pro"),
                "evaluation_count": len(self.evaluation_history["pro"])
            },
            "con": {
                "reward_bank": self.con_reward_bank,
                "punishment_debt": self.con_punishment_debt,
                "net_score": self.get_net_performance("con"),
                "evaluation_count": len(self.evaluation_history["con"])
            }
        }

class EnhancedStaminaManager:
    """Enhanced stamina manager with reward/punishment integration"""
    
    def __init__(self, initial_stamina: int = 27, rules_loader: ReasoningRulesLoader = None):
        self.initial_stamina = initial_stamina
        self.pro_stamina = initial_stamina
        self.con_stamina = initial_stamina
        # Use rule costs from the loader
        if rules_loader:
            self.rule_costs = rules_loader.get_rule_costs()
        else:
            self.rule_costs = {}
        # Adjusted stamina thresholds for synthesis mode (fit new scale: 27)
        self.synthesis_threshold = 18  # Start synthesis when either side drops below this
        self.critical_threshold = 12   # Critical synthesis mode
        self.emergency_threshold = 6   # Emergency conclusion mode
    
    def should_trigger_synthesis(self) -> Tuple[bool, str]:
        """Check if synthesis mode should be triggered"""
        if self.pro_stamina <= self.emergency_threshold or self.con_stamina <= self.emergency_threshold:
            return True, "emergency"
        elif self.pro_stamina <= self.critical_threshold or self.con_stamina <= self.critical_threshold:
            return True, "critical"
        elif self.pro_stamina <= self.synthesis_threshold or self.con_stamina <= self.synthesis_threshold:
            return True, "standard"
        else:
            return False, "none"
    
    def apply_reward_exchange(self, side: str, stamina_gain: int):
        """Apply stamina gain from reward exchange"""
        if side == "pro":
            self.pro_stamina = min(self.initial_stamina, self.pro_stamina + stamina_gain)
        else:
            self.con_stamina = min(self.initial_stamina, self.con_stamina + stamina_gain)
    
    def apply_punishment_loss(self, side: str, stamina_loss: int):
        """Apply stamina loss from punishment debt"""
        if side == "pro":
            self.pro_stamina = max(0, self.pro_stamina - stamina_loss)
        else:
            self.con_stamina = max(0, self.con_stamina - stamina_loss)
    
    def can_afford_rules(self, side: str, rules: List[Dict]) -> Tuple[List[Dict], int]:
        """Check which rules can be afforded and return cost"""
        current_stamina = self.pro_stamina if side == "pro" else self.con_stamina
        affordable_rules = []
        total_cost = 0
        
        # Sort rules by relevance score (higher first)
        sorted_rules = sorted(rules, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        for rule in sorted_rules:
            rule_id = rule.get('rule_id', '')
            rule_cost = self.rule_costs.get(rule_id, 2)  # Default cost 2
            if current_stamina - total_cost >= rule_cost:
                affordable_rules.append(rule)
                total_cost += rule_cost
                
                # Limit to maximum 3 rules per turn
                if len(affordable_rules) >= 3:
                    break
        
        return affordable_rules, total_cost
    
    def spend_stamina(self, side: str, cost: int) -> bool:
        """Spend stamina and return if successful"""
        if side == "pro":
            if self.pro_stamina >= cost:
                self.pro_stamina -= cost
                return True
        else:
            if self.con_stamina >= cost:
                self.con_stamina -= cost
                return True
        return False
    
    def can_afford_basic_argument(self, side: str) -> bool:
        """Check if side can afford a basic argument"""
        current_stamina = self.pro_stamina if side == "pro" else self.con_stamina
        return current_stamina >= 1
    
    def is_exhausted(self, side: str) -> bool:
        """Check if side is completely exhausted"""
        current_stamina = self.pro_stamina if side == "pro" else self.con_stamina
        return current_stamina < 1
    
    def both_exhausted(self) -> bool:
        """Check if both sides are exhausted"""
        return self.is_exhausted("pro") and self.is_exhausted("con")
    
    def get_status(self) -> Dict:
        """Get current stamina status"""
        synthesis_trigger, synthesis_level = self.should_trigger_synthesis()
        
        return {
            "pro_stamina": self.pro_stamina,
            "con_stamina": self.con_stamina,
            "pro_exhausted": self.is_exhausted("pro"),
            "con_exhausted": self.is_exhausted("con"),
            "both_exhausted": self.both_exhausted(),
            "synthesis_trigger": synthesis_trigger,
            "synthesis_level": synthesis_level
        }

class ComplexityAnalyzer:
    """Analyzes question complexity to guide reasoning effort"""
    
    def __init__(self):
        # Keywords that indicate different complexity levels
        self.complexity_indicators = {
            "simple": {
                "keywords": ["what is", "define", "when", "where", "who", "basic", "simple", "true or false", "yes or no"],
                "patterns": ["is", "are", "was", "were"],
                "math_simple": ["add", "subtract", "multiply", "divide", "plus", "minus", "times"]
            },
            "moderate": {
                "keywords": ["how", "why", "explain", "describe", "compare", "contrast", "analyze", "evaluate"],
                "patterns": ["should", "could", "might", "would"],
                "math_moderate": ["percentage", "ratio", "average", "mean", "probability"]
            },
            "complex": {
                "keywords": ["synthesis", "integration", "framework", "paradigm", "philosophy", "existential", "consciousness"],
                "patterns": ["if...then", "what if", "suppose", "consider"],
                "abstract_concepts": ["justice", "truth", "reality", "meaning", "purpose", "free will", "determinism"]
            },
            "very_complex": {
                "keywords": ["paradox", "dialectical", "meta", "recursive", "emergent", "non-linear", "quantum", "relativistic"],
                "patterns": ["both...and", "neither...nor", "not only...but also"],
                "high_level": ["epistemology", "ontology", "phenomenology", "hermeneutics", "deconstruction"]
            }
        }
    
    def assess_complexity(self, question: str) -> Tuple[str, float, Dict]:
        """Assess question complexity and return level, confidence, and reasoning"""
        
        question_lower = question.lower()
        word_count = len(question.split())
        
        # Score each complexity level
        scores = {"simple": 0, "moderate": 0, "complex": 0, "very_complex": 0}
        
        for level, indicators in self.complexity_indicators.items():
            # Keyword matching
            for keyword in indicators.get("keywords", []):
                if keyword in question_lower:
                    scores[level] += 2
            
            # Pattern matching
            for pattern in indicators.get("patterns", []):
                if pattern in question_lower:
                    scores[level] += 1.5
            
            # Special category matching
            for category in ["math_simple", "math_moderate", "abstract_concepts", "high_level"]:
                for term in indicators.get(category, []):
                    if term in question_lower:
                        scores[level] += 2.5
        
        # Length-based adjustments
        if word_count < 8:
            scores["simple"] += 1
        elif word_count > 20:
            scores["complex"] += 1
            scores["very_complex"] += 0.5
        
        # Question mark patterns
        question_marks = question.count("?")
        if question_marks > 1:
            scores["complex"] += 1
        
        # Mathematical expressions
        math_symbols = any(symbol in question for symbol in ["+", "-", "*", "/", "=", "^", "√", "∫", "∑"])
        if math_symbols:
            if any(term in question_lower for term in ["prove", "theorem", "lemma", "axiom"]):
                scores["very_complex"] += 3
            elif any(term in question_lower for term in ["solve", "calculate", "find"]):
                scores["moderate"] += 2
        
        # Determine final complexity
        max_score = max(scores.values())
        if max_score == 0:
            complexity_level = "moderate"  # Default
            confidence = 0.5
        else:
            complexity_level = max(scores.keys(), key=scores.get)
            confidence = min(1.0, max_score / 5.0)  # Normalize to 0-1
        
        reasoning = {
            "word_count": word_count,
            "scores": scores,
            "question_marks": question_marks,
            "has_math": math_symbols,
            "detected_indicators": self._get_detected_indicators(question_lower)
        }
        
        return complexity_level, confidence, reasoning
    
    def _get_detected_indicators(self, question_lower: str) -> List[str]:
        """Get list of complexity indicators found in question"""
        found = []
        for level, indicators in self.complexity_indicators.items():
            for category, terms in indicators.items():
                for term in terms:
                    if term in question_lower:
                        found.append(f"{level}:{term}")
        return found

class EnhancedReasoningRulesRAG:
    """Enhanced RAG system with synthesis-aware rule selection using JSON rules"""
    
    def __init__(self, rules_file_path: str = "reasoning_rules.json"):
        print("🔍 Initializing Enhanced Reasoning Rules RAG System...")
        self.embedder = SentenceTransformer('all-mpnet-base-v2')
        self.complexity_analyzer = ComplexityAnalyzer()
        self.rules_loader = ReasoningRulesLoader(rules_file_path)
        self.reasoning_rules = self.rules_loader.get_all_rules()
        
        self.build_vector_database()
    
    def build_vector_database(self):
        """Build FAISS vector database for rule retrieval"""
        print("Building vector database...")
        texts = []
        rule_ids = []
        
        for rule_id, rule_data in self.reasoning_rules.items():
            texts.append(rule_data["rule"])
            rule_ids.append(rule_id)
            
            texts.append(f"{rule_data['rule']} Framework: {rule_data['framework']}")
            rule_ids.append(rule_id)
            
            for keyword in rule_data.get("keywords", []):
                texts.append(f"When considering '{keyword}', apply: {rule_data['rule']}")
                rule_ids.append(rule_id)
        
        with suppress_stdout_stderr():
            embeddings = self.embedder.encode(texts, show_progress_bar=False)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        self.texts = texts
        self.rule_ids = rule_ids
        print(f"✅ Vector database built with {len(texts)} entries")
    
    def reload_rules(self):
        """Reload rules from file and rebuild database"""
        print("🔄 Reloading rules from file...")
        self.rules_loader.reload_rules()
        self.reasoning_rules = self.rules_loader.get_all_rules()
        self.build_vector_database()
    
    def retrieve_relevant_rules(self, query: str, top_k: int = 5, prefer_synthesis: bool = False,
                              memory_summary: str = "",
                              side: str = None,
                              stamina: int = None,
                              memory_graph: 'DebateMemoryGraph' = None,
                              reflection_context: str = None,
                              mentor_reflection: str = None) -> List[Dict]:
        """Retrieve most relevant reasoning rules with dynamic selection based on stamina, depth, and past effectiveness. Always include robust awareness rules."""
        # Assess question complexity
        complexity_level, complexity_confidence, complexity_reasoning = self.complexity_analyzer.assess_complexity(query)
        print(f"🧠 Question complexity: {complexity_level.upper()} (confidence: {complexity_confidence:.2f})")
        # Enhance query with memory context for better rule selection
        enhanced_query = query
        if memory_summary:
            enhanced_query += f" Context: {memory_summary}"
        with suppress_stdout_stderr():
            query_embedding = self.embedder.encode([enhanced_query], show_progress_bar=False)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k * 4)
        results = []
        seen_rules = set()
        # Reference frame rules (Thousand Brains Theory inspired)
        reference_frame_rules = [
            'virtue_ethics_frame', 'consequentialist_frame', 'deontological_frame', 'phenomenological_frame',
            'systems_thinking_frame', 'pragmatic_frame', 'existentialist_frame', 'scientific_empirical_frame',
            'narrative_interpretive_frame', 'critical_theory_frame'
        ]
        synthesis_rules = [
            'framework_integration', 'dynamic_synthesis', 'perspective_integration', 'synthesize_perspectives', 'frame_blending'
        ]
        meta_rules = ['meta_reasoning_self_critique', 'argument_decomposition']
        # Always include at least two reference frame rules and one synthesis rule for moderate+ complexity
        forced_rules = set()
        if complexity_level in ['moderate', 'complex', 'very_complex']:
            for rule_id in reference_frame_rules[:]:
                if rule_id in self.reasoning_rules:
                    forced_rules.add(rule_id)
                    if len(forced_rules) >= 2:
                        break
            for rule_id in synthesis_rules:
                if rule_id in self.reasoning_rules:
                    forced_rules.add(rule_id)
                    break
            for rule_id in meta_rules:
                if rule_id in self.reasoning_rules:
                    forced_rules.add(rule_id)
        # Always include meta-reasoning for all but simple
        if complexity_level != 'simple':
            for rule_id in meta_rules:
                if rule_id in self.reasoning_rules:
                    forced_rules.add(rule_id)
        # Add forced rules to results first
        for rule_id in forced_rules:
            rule_data = self.reasoning_rules[rule_id]
            results.append({
                "rule_id": rule_id,
                "rule_text": rule_data["rule"],
                "framework": rule_data["framework"],
                "relevance_score": 1.2,
                "complexity": rule_data.get("complexity", "medium"),
                "question_complexity": complexity_level,
                "complexity_confidence": complexity_confidence
            })
            seen_rules.add(rule_id)
        
        # If synthesis preferred, boost synthesis-related rules
        synthesis_rules = ["dialectical_tension", "perspective_integration", "dynamic_synthesis", "framework_integration"]
        
        # Complexity-based rule filtering
        complexity_appropriate_rules = self._get_complexity_appropriate_rules(complexity_level)
        
        # Memory-based rule boosts
        memory_aware_rules = ["situational_awareness"]
        
        for score, idx in zip(scores[0], indices[0]):
            rule_id = self.rule_ids[idx]
            if rule_id not in seen_rules and len(results) < top_k * 2:
                rule_data = self.reasoning_rules[rule_id]
                
                # Boost synthesis rules when needed
                adjusted_score = float(score)
                if prefer_synthesis and rule_id in synthesis_rules:
                    adjusted_score += 0.15
                
                # Boost complexity-appropriate rules
                if rule_id in complexity_appropriate_rules:
                    adjusted_score += 0.2
                
                # Force include complexity_awareness for complex questions
                if complexity_level in ["complex", "very_complex"] and rule_id == "complexity_awareness":
                    adjusted_score += 0.5
                
                # Always include emotional_awareness if emotional keywords detected
                emotional_keywords = ["feel", "emotion", "values", "care", "frustration", "anger", "fear", "hope"]
                if any(keyword in query.lower() for keyword in emotional_keywords) and rule_id == "emotional_awareness":
                    adjusted_score += 0.15
                
                # Strong boost for situational_awareness when memory context exists
                if memory_summary and rule_id in memory_aware_rules:
                    adjusted_score += 0.40
                
                # Boost situational_awareness for questions about strategy, adaptation, or patterns
                situational_keywords = [
                    "strategy", "pattern", "adapt", "learn", "previous", "before", "context", "flow",
                    "opponent", "argument", "debate", "round", "turn", "history", "memory", "conversation",
                    "response", "building", "foundation", "improvement", "evolution", "progression",
                    "effectiveness", "performance", "quality", "trend", "change", "development", "synthesis"
                ]
                if any(keyword in query.lower() for keyword in situational_keywords) and rule_id == "situational_awareness":
                    adjusted_score += 0.3
                
                results.append({
                    "rule_id": rule_id,
                    "rule_text": rule_data["rule"],
                    "framework": rule_data["framework"],
                    "relevance_score": adjusted_score,
                    "complexity": rule_data.get("complexity", "medium"),
                    "question_complexity": complexity_level,
                    "complexity_confidence": complexity_confidence
                })
                seen_rules.add(rule_id)
        
        # Auto-include critical rules for complex debates
        if complexity_level in ["simple","moderate", "complex", "very_complex"]:
            # Force include situational awareness if not already present
            if "situational_awareness" not in [r['rule_id'] for r in results]:
                situational_rule = {
                    "rule_id": "situational_awareness",
                    "rule_text": self.reasoning_rules["situational_awareness"]["rule"],
                    "framework": self.reasoning_rules["situational_awareness"]["framework"],
                    "relevance_score": 0.9,
                    "complexity": "medium",
                    "question_complexity": complexity_level,
                    "complexity_confidence": complexity_confidence
                }
                results.append(situational_rule)
            
            # Force include complexity awareness if not already present
            if "complexity_awareness" not in [r['rule_id'] for r in results]:
                complexity_rule = {
                    "rule_id": "complexity_awareness",
                    "rule_text": self.reasoning_rules["complexity_awareness"]["rule"],
                    "framework": self.reasoning_rules["complexity_awareness"]["framework"],
                    "relevance_score": 0.9,
                    "complexity": "low",
                    "question_complexity": complexity_level,
                    "complexity_confidence": complexity_confidence
                }
                results.append(complexity_rule)
            
            if "emotional_awareness" not in [r['rule_id'] for r in results]:
                emotional_rule = {
                    "rule_id": "emotional_awareness",
                    "rule_text": self.reasoning_rules["emotional_awareness"]["rule"],
                    "framework": self.reasoning_rules["emotional_awareness"]["framework"],
                    "relevance_score": 0.9,
                    "complexity": "medium",
                    "question_complexity": complexity_level,
                    "complexity_confidence": complexity_confidence
                }
                results.append(emotional_rule)
            
            if "link_awareness" not in [r['rule_id'] for r in results]:
                link_rule = {
                    "rule_id": "link_awareness",
                    "rule_text": self.reasoning_rules["link_awareness"]["rule"],
                    "framework": self.reasoning_rules["link_awareness"]["framework"],
                    "relevance_score": 0.9,
                    "complexity": "medium",
                    "question_complexity": complexity_level,
                    "complexity_confidence": complexity_confidence
                }
                results.append(link_rule)

        
        # Auto-include situational awareness when memory context exists
        if memory_summary and "situational_awareness" and "link_awareness" not in [r['rule_id'] for r in results]:
            situational_rule = {
                "rule_id": "situational_awareness",
                "rule_text": self.reasoning_rules["situational_awareness"]["rule"],
                "framework": self.reasoning_rules["situational_awareness"]["framework"],
                "relevance_score": 0.8,
                "complexity": "medium",
                "question_complexity": complexity_level,
                "complexity_confidence": complexity_confidence
            },
            link_role={
                 "rule_id": "link_awareness",
                "rule_text": self.reasoning_rules["link_awareness"]["rule"],
                "framework": self.reasoning_rules["link_awareness"]["framework"],
                "relevance_score": 0.9,
                "complexity": "medium",
                "question_complexity": complexity_level,
                "complexity_confidence": complexity_confidence
            }
            results.append(situational_rule)
            results.append(link_role)
            print("📊 AUTO-INCLUDED: situational_awareness due to memory context")
        
        # --- Robust Awareness Rule Inclusion ---
        awareness_rules = [
            'situational_awareness', 'complexity_awareness', 'emotional_awareness', 'link_awareness',
            'contextual_sensitivity', 'meta_reasoning_self_critique'
        ]
        # Improved emotional and situational keyword detection
        emotional_keywords = [
            "feel", "emotion", "values", "care", "frustration", "anger", "fear", "hope", "sad", "joy", "passion", "hurt", "love", "hate", "stress", "worry", "anxiety", "excited", "relief", "resent", "envy", "pride", "shame", "guilt", "regret", "affect", "sentiment", "mood", "psychology", "trauma", "wellbeing", "mental", "happiness", "suffering"
        ]
        situational_keywords = [
            "strategy", "pattern", "adapt", "learn", "previous", "before", "context", "flow", "opponent", "argument", "debate", "round", "turn", "history", "memory", "conversation", "response", "building", "foundation", "improvement", "evolution", "progression", "effectiveness", "performance", "quality", "trend", "change", "development", "synthesis", "situation", "circumstance", "timing", "sequence", "phase", "stage"
        ]
        link_keywords = [
            "link", "relationship", "connection", "network", "cluster", "bridge", "pattern", "topology", "graph", "interact", "interconnection", "association", "correlation", "dependency", "structure", "web", "system", "map", "connect", "tie", "bond", "related", "interrelated"
        ]
        # Always include and boost awareness rules
        for rule_id in awareness_rules:
            if rule_id not in [r['rule_id'] for r in results]:
                if rule_id in self.reasoning_rules:
                    rule_data = self.reasoning_rules[rule_id]
                    results.append({
                        "rule_id": rule_id,
                        "rule_text": rule_data["rule"],
                        "framework": rule_data["framework"],
                        "relevance_score": 1.0,
                        "complexity": rule_data.get("complexity", "medium"),
                        "question_complexity": complexity_level,
                        "complexity_confidence": complexity_confidence
                    })
        # Boost scores for awareness rules if detected in question
        q_lower = query.lower()
        for r in results:
            if r['rule_id'] == 'emotional_awareness' and any(k in q_lower for k in emotional_keywords):
                r['relevance_score'] += 0.5
            if r['rule_id'] == 'situational_awareness' and any(k in q_lower for k in situational_keywords):
                r['relevance_score'] += 0.5
            if r['rule_id'] == 'link_awareness' and any(k in q_lower for k in link_keywords):
                r['relevance_score'] += 0.5
            if r['rule_id'] == 'contextual_sensitivity' and ("context" in q_lower or "culture" in q_lower or "environment" in q_lower):
                r['relevance_score'] += 0.3
            if r['rule_id'] == 'meta_reasoning_self_critique' and ("assumption" in q_lower or "bias" in q_lower or "critique" in q_lower or "reflect" in q_lower):
                r['relevance_score'] += 0.3
        
        # Sort and limit to top_k
        sorted_results = sorted(results, key=lambda x: x['relevance_score'], reverse=True)
        # --- DYNAMIC RULE SELECTION ---
        # If side, stamina, and memory_graph are provided, further filter and score rules
        if side and stamina is not None and memory_graph is not None:
            # 1. Prefer lower-cost rules if stamina is low
            stamina_threshold = 10
            if stamina <= stamina_threshold:
                sorted_results = sorted(sorted_results, key=lambda x: self.rules_loader.rule_costs.get(x['rule_id'], 2))
            # 2. Prefer higher-complexity rules for deep questions
            complexity_level = sorted_results[0]['question_complexity'] if sorted_results else 'moderate'
            if complexity_level in ['complex', 'very_complex']:
                sorted_results = sorted(sorted_results, key=lambda x: {'low':0, 'medium':1, 'high':2, 'very_high':3}.get(x['complexity'],1), reverse=True)
            # 3. Prefer rules with higher avg quality for this side (from memory graph)
            rule_effectiveness = memory_graph.get_rule_effectiveness(side)
            rule_quality_map = {r[0]: r[1]['avg_quality'] for r in rule_effectiveness['best_rules']}
            def rule_score(rule):
                return rule_quality_map.get(rule['rule_id'], 0)
            sorted_results = sorted(sorted_results, key=rule_score, reverse=True)
        # Chain sort keys for efficiency
        sorted_results = sorted_results[:top_k]
        
        # Always include and boost temporal and sensorimotor rules
        temporal_rules = ['temporal_dynamics', 'consequence_analysis']
        sensorimotor_rules = ['sensorimotor_reasoning', 'spatial_analogy']
        for rule_id in temporal_rules + sensorimotor_rules:
            if rule_id in self.reasoning_rules and rule_id not in {r['rule_id'] for r in results}:
                rule_data = self.reasoning_rules[rule_id]
                results.append({
                    "rule_id": rule_id,
                    "rule_text": rule_data["rule"],
                    "framework": rule_data["framework"],
                    "relevance_score": 1.0,
                    "complexity": rule_data.get("complexity", "medium"),
                    "question_complexity": complexity_level,
                    "complexity_confidence": complexity_confidence
                })
        # Boost temporal/sensorimotor/context rules if detected in question
        q_lower = query.lower()
        if any(k in q_lower for k in ["time", "sequence", "trend", "consequence", "future", "history", "change", "progression", "cycle", "feedback", "effect", "result", "outcome", "implication", "impact", "ripple", "cascade", "chain reaction"]):
            for r in results:
                if r['rule_id'] in temporal_rules:
                    r['relevance_score'] += 0.5
        if any(k in q_lower for k in ["spatial", "physical", "movement", "embodied", "analogy", "simulation", "imagine", "mechanism", "force", "structure", "shape", "path", "geometry", "fit", "arrangement", "layout", "map", "distance", "direction"]):
            for r in results:
                if r['rule_id'] in sensorimotor_rules:
                    r['relevance_score'] += 0.5
        # For simple questions, prefer direct/low-complexity rules
        if complexity_level == 'simple':
            results = sorted(results, key=lambda x: {'low':0, 'medium':1, 'high':2, 'very_high':3}.get(x['complexity'],1))
        # For complex/very_complex, prefer high-complexity, temporal, sensorimotor, and awareness rules
        if complexity_level in ['complex', 'very_complex']:
            for r in results:
                if r['rule_id'] in temporal_rules + sensorimotor_rules + ['situational_awareness', 'contextual_sensitivity', 'meta_reasoning_self_critique']:
                    r['relevance_score'] += 0.5
            results = sorted(results, key=lambda x: {'low':0, 'medium':1, 'high':2, 'very_high':3}.get(x['complexity'],1), reverse=True)
        
        # --- ENHANCED: Strongly boost new/rewritten rules for relevant questions ---
        emotional_rules = ['emotional_awareness', 'emotional_regulation', 'collective_emotional_dynamics']
        bias_rules = ['cognitive_bias_awareness']
        context_rules = ['situational_awareness', 'contextual_memory_integration']
        thread_rules = ['argument_threading', 'thematic_continuity']
        frame_rules = ['frame_blending', 'reference_frame_switching']
        # Boost for emotional/group questions
        if any(k in q_lower for k in ["emotion", "feeling", "group", "collective", "sentiment", "mood", "psychology", "bias", "regulation", "anger", "joy", "fear", "hope", "public", "crowd", "groupthink"]):
            for r in results:
                if r['rule_id'] in emotional_rules + bias_rules:
                    r['relevance_score'] += 0.7
        # Boost for context/memory
        if any(k in q_lower for k in ["context", "memory", "history", "previous", "thread", "theme", "continuity", "pattern", "sequence", "connect", "build on", "carry forward"]):
            for r in results:
                if r['rule_id'] in context_rules + thread_rules:
                    r['relevance_score'] += 0.7
        # Boost for synthesis/group/complex
        if prefer_synthesis or complexity_level in ['complex', 'very_complex']:
            for r in results:
                if r['rule_id'] in frame_rules + ['collective_emotional_dynamics', 'contextual_memory_integration']:
                    r['relevance_score'] += 0.7
        
        # --- ADAPTIVE LOGIC: Use reflection context to adjust rule selection ---
        reflection_boosts = {}
        if reflection_context:
            rc = reflection_context.lower()
            # Example: If reflection says 'need more creativity', boost creativity rules
            if 'creativity' in rc or 'novel' in rc or 'original' in rc:
                for rule in ['divergent_thinking', 'analogical_creativity', 'serendipity_cultivation', 'combinatorial_creativity']:
                    reflection_boosts[rule] = reflection_boosts.get(rule, 0) + 0.7
            if 'repetitive' in rc or 'repetition' in rc or 'same' in rc:
                for rule in ['argument_threading', 'thematic_continuity', 'meta_reasoning_self_critique']:
                    reflection_boosts[rule] = reflection_boosts.get(rule, 0) + 0.7
                # Optionally, suppress rules that have been overused (from memory)
            if 'evidence' in rc or 'proof' in rc or 'study' in rc:
                for rule in ['evidence_hierarchies', 'bayesian_reasoning', 'authority_evaluation_advanced']:
                    reflection_boosts[rule] = reflection_boosts.get(rule, 0) + 0.7
            if 'context' in rc or 'missed context' in rc or 'awareness' in rc:
                for rule in ['situational_awareness', 'contextual_sensitivity', 'contextual_memory_integration']:
                    reflection_boosts[rule] = reflection_boosts.get(rule, 0) + 0.7
            if 'synthesis' in rc or 'integration' in rc or 'combine' in rc:
                for rule in ['dynamic_synthesis', 'perspective_integration', 'frame_blending', 'synthesize_perspectives']:
                    reflection_boosts[rule] = reflection_boosts.get(rule, 0) + 0.7
            if 'emotion' in rc or 'group' in rc or 'collective' in rc:
                for rule in ['emotional_awareness', 'collective_emotional_dynamics', 'emotional_regulation']:
                    reflection_boosts[rule] = reflection_boosts.get(rule, 0) + 0.7
            if 'bias' in rc or 'assumption' in rc:
                for rule in ['cognitive_bias_awareness', 'probe_hidden_assumptions', 'meta_reasoning_self_critique']:
                    reflection_boosts[rule] = reflection_boosts.get(rule, 0) + 0.7
            # If reflection says 'opponent is strong', boost opponent modeling
            if 'opponent' in rc or 'counter' in rc or 'steelman' in rc:
                for rule in ['opponent_modeling', 'challenge_framework']:
                    reflection_boosts[rule] = reflection_boosts.get(rule, 0) + 0.7
        # Mentor meta-reflection can also boost/suppress
        if mentor_reflection:
            mc = mentor_reflection.lower()
            if 'weakness' in mc or 'improve' in mc:
                for rule in ['meta_reasoning_self_critique', 'argument_decomposition', 'thematic_continuity']:
                    reflection_boosts[rule] = reflection_boosts.get(rule, 0) + 0.7
            if 'strength' in mc or 'continue' in mc:
                for rule in ['argument_threading', 'frame_blending', 'dynamic_synthesis']:
                    reflection_boosts[rule] = reflection_boosts.get(rule, 0) + 0.7
        # Apply boosts to results
        for r in results:
            if r['rule_id'] in reflection_boosts:
                r['relevance_score'] += reflection_boosts[r['rule_id']]
        
        return sorted_results
    
    def _get_complexity_appropriate_rules(self, complexity_level: str) -> List[str]:
        """Get rules appropriate for the given complexity level"""
        
        rule_complexity_map = {
            "simple": [
                "complexity_awareness", "evidence_hierarchies", "generalization_boundaries", 
                "analogical_reasoning", "situational_awareness"
            ],
            "moderate": [
                "complexity_awareness", "causation_analysis", "bias_recognition_advanced", 
                "authority_evaluation_advanced", "temporal_dynamics", "contextual_sensitivity", 
                "emotional_awareness", "situational_awareness", "modal_reasoning", "abductive_reasoning"
            ],
            "complex": [
                "complexity_awareness", "metacognitive_monitoring", "reasoning_strategy_selection", 
                "dialectical_tension", "perspective_integration", "multi_order_consequences", 
                "emotional_awareness", "situational_awareness", "dynamic_synthesis", 
                "cognitive_load_management", "systems_interaction"
            ],
            "very_complex": [
                "complexity_awareness", "systems_interaction", "recursive_reasoning", 
                "framework_integration", "reasoning_limits_recognition", "dialectical_tension", 
                "dynamic_synthesis", "situational_awareness", "perspective_integration",
                "metacognitive_monitoring", "multi_order_consequences"
            ]
        }
        
        return rule_complexity_map.get(complexity_level, rule_complexity_map["moderate"])
    
    def emergent_synthesis_frame(self, query: str, top_k: int = 4, memory_summary: str = "", creativity_boost: float = 0.5, memory_graph=None) -> dict:
        """Thousand Brains Theory-inspired: Simulate multiple independent reasoning columns, blend their outputs, and form a novel, integrative reasoning frame for creativity and insight."""
        # Retrieve a diverse set of top rules (prefer diversity in complexity and type)
        rules = self.retrieve_relevant_rules(query, top_k=top_k * 2, memory_summary=memory_summary)
        if not rules:
            return {"frame": "No emergent synthesis possible.", "rules": []}
        # Select rules to maximize diversity (by complexity/type)
        selected = []
        seen_types = set()
        for r in rules:
            c = r.get('complexity', 'medium')
            if c not in seen_types:
                selected.append(r)
                seen_types.add(c)
            if len(selected) >= top_k:
                break
        if len(selected) < top_k:
            selected = rules[:top_k]
        # Track and boost underused frameworks for exploration
        if memory_graph:
            analytics = memory_graph.get_frame_analytics()
            used_frames = {a['frame'] for a in analytics}
            all_rule_ids = set(self.reasoning_rules.keys())
            never_used = all_rule_ids - used_frames
            # Boost never-used and underused rules
            for r in rules:
                if r['rule_id'] in never_used:
                    r['relevance_score'] += 1.0  # Strong boost for never-used
                elif r['rule_id'] not in used_frames:
                    r['relevance_score'] += 0.5
        # Simulate independent columns (frames)
        columns = []
        for idx, rule in enumerate(selected):
            columns.append(f"[Column {idx+1}] {rule['rule_text']}\n  Framework: {rule['framework']}")
        # Blend outputs into a higher-level synthesis
        synthesis = (
            "Thousand Brains Emergent Synthesis Frame:\n"
            "- Multiple independent reasoning columns (frames) are simulated in parallel, each applying a distinct rule or perspective.\n"
            "- Their outputs are then blended, compared, and integrated to form a novel, creative, and robust approach.\n"
            "- This process encourages diversity, even conflict, among reasoning strategies, leading to richer synthesis and new insights.\n"
            "- Use this frame to approach the problem from multiple angles, then actively integrate the best elements into your argument.\n"
            "- IMPORTANT: Go beyond simply combining or stitching together the columns. Seek to generate fundamentally new concepts, analogies, or creative leaps that are not present in any single frame. Look for emergent patterns, surprising connections, or original insights that would not arise from any one column alone.\n"
            f"\nINDEPENDENT COLUMNS:\n" + "\n".join(columns) +
            "\n\nINTEGRATIVE SYNTHESIS INSTRUCTION:\nBlend the above columns' insights, but explicitly attempt to create a new, emergent idea, analogy, or synthesis that transcends the sum of the parts. Surprise the reader with a creative leap or original perspective."
        )
        return {"frame": synthesis, "rules": selected}
    
    def generate_columns_and_synthesis(self, prompt: str, columns: list, side: str, stamina: int, synthesis_level: str, rl_status: dict, memory_summary: str = "") -> dict:
        """Generate sub-arguments for each column, then synthesize them into a final argument."""
        sub_arguments = []
        for idx, col in enumerate(columns):
            col_prompt = f"COLUMN {idx+1} FRAME:\n{col['rule_text']}\nFramework: {col['framework']}\n\nPrompt: {prompt}\nGenerate a sub-argument from this frame's perspective."
            debater = OllamaDebater(side=side)
            sub_arg = debater.generate_basic_argument(col_prompt, stamina, synthesis_level, rl_status, memory_summary)
            sub_arguments.append(sub_arg)
        # Synthesize
        synth_prompt = f"SYNTHESIS TASK:\nGiven these sub-arguments from different frames:\n" + "\n---\n".join(sub_arguments) + "\n\nIntegrate their insights into a single, creative, novel argument that combines the best of each."
        debater = OllamaDebater(side=side)
        final_argument = debater.generate_basic_argument(synth_prompt, stamina, synthesis_level, rl_status, memory_summary)
        return {"sub_arguments": sub_arguments, "final_argument": final_argument}

from mentor_module import EnhancedMentorEvaluator

class OllamaDebater:
    """Enhanced debater with strict RL awareness and memory integration"""
    
    def __init__(self, model_name="llama3.2:3b", side="pro"):
        self.model_name = model_name
        self.side = side
        self.base_url = "http://localhost:11434"
        
    def generate_argument_with_stamina(self, prompt: str, rules: List[Dict], stamina: int, 
                                     synthesis_level: str, rl_status: Dict, memory_summary: str = "") -> str:
        """Generate argument with stamina, strict RL awareness, and memory integration"""
        
        rules_text = "\n".join([f"- {rule['rule_text']} (Cost: {rule.get('complexity', 'medium')})" for rule in rules])
        frameworks = "\n".join([f"  {rule.get('framework', '')}" for rule in rules])
        # Add linguistic fluidity rules if present
        extra_fluidity = ""
        for rule in rules:
            if rule.get('rule_id') == 'natural_expression':
                extra_fluidity += "\n- Express your argument in a natural, conversational, and relatable style."
            if rule.get('rule_id') == 'emotional_resonance':
                extra_fluidity += "\n- Use appropriate emotional elements to make your argument more compelling and human."
        if not extra_fluidity:
            extra_fluidity = "\n- Use clear, engaging, and human language. Avoid sounding robotic or overly academic."
        
        # Get question complexity information
        question_complexity = rules[0].get('question_complexity', 'moderate') if rules else 'moderate'
        complexity_guidance = self._get_complexity_guidance(question_complexity)
        
        # Memory integration
        memory_context = ""
        if memory_summary:
            memory_context = f"\n🧠 MEMORY CONTEXT: {memory_summary}\nUse this to inform your strategy and avoid repeating ineffective patterns."
        
        # Synthesis urgency based on level
        synthesis_instructions = {
            "emergency": "EMERGENCY MODE: MUST REACH CONCLUSION NOW! Focus entirely on synthesis and resolution!",
            "critical": "CRITICAL MODE: Prioritize conclusion-oriented reasoning! Move toward resolution!",
            "standard": "SYNTHESIS MODE: Begin considering how to integrate perspectives and reach conclusions.",
            "none": "Build strong reasoning foundation for eventual synthesis."
        }
        
        synthesis_instruction = synthesis_instructions.get(synthesis_level, "")
        
        # RL performance warnings
        rl_warning = ""
        if rl_status.get('punishment_debt', 0) > 10:
            rl_warning = f"\n⚠️ WARNING: {rl_status['punishment_debt']} punishment debt! Use high-quality reasoning to avoid further penalties!"
        elif rl_status.get('reward_bank', 0) >= 3:
            rl_warning = f"\n✨ REWARD OPPORTUNITY: {rl_status['reward_bank']} reward points available for stamina exchange! (3 rewards = 9 stamina)"
        
        enhanced_prompt = f"""STRICT RL REASONING SYSTEM WITH MEMORY - QUALITY ENFORCEMENT
REMAINING STAMINA: {stamina}
SYNTHESIS LEVEL: {synthesis_level.upper()}
QUESTION COMPLEXITY: {question_complexity.upper()}
{complexity_guidance}
{synthesis_instruction}{rl_warning}{memory_context}

PERFORMANCE STATUS:
- Reward Bank: {rl_status.get('reward_bank', 0)} points
- Punishment Debt: {rl_status.get('punishment_debt', 0)} points
- Net Score: {rl_status.get('net_score', 0)}

AVAILABLE REASONING FRAMEWORKS (USE WISELY):
{rules_text}

CRITICAL THINKING QUESTIONS:
{frameworks}

ROLE: {self.side.title()}-side debater
CONTEXT: {prompt}

⚠️ QUALITY REQUIREMENTS:
- Poor reasoning = -8 to -15 punishment points + stamina loss
- Good reasoning = +1 to +3 reward points (3 rewards = 9 stamina!)
- Focus on conclusion-oriented reasoning when stamina is low
- Avoid bad arguments at all costs - penalties are severe!
- Learn from your memory context to improve performance
- {complexity_guidance}
{extra_fluidity}

Generate HIGH-QUALITY argument (3-5 sentences) that demonstrates sophisticated reasoning, memory awareness, and a human, conversational tone:"""
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": enhanced_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "num_predict": 600,
                    "stop": ["STAMINA:", "PERFORMANCE:", "USER:", "HUMAN:", "🎓", "Mentor:", "QUALITY:"],
                    "repeat_penalty": 1.1,
                    "num_ctx": 4096
                }
            }
            
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "Error").strip()
                return self._clean_response(answer)
            else:
                return f"Error: {response.status_code}"
                
        except Exception as e:
            return f"Error: {str(e)[:50]}"
    
    def generate_basic_argument(self, prompt: str, stamina: int, synthesis_level: str, rl_status: Dict, memory_summary: str = "") -> str:
        """Generate basic argument with RL awareness and memory context"""
        
        synthesis_urgency = ""
        if synthesis_level in ["emergency", "critical"]:
            synthesis_urgency = f"\n{synthesis_level.upper()} MODE: Focus on reaching conclusion!"
        
        rl_warning = ""
        if rl_status.get('punishment_debt', 0) > 5:
            rl_warning = f"\nWARNING: {rl_status['punishment_debt']} punishment debt! Avoid poor reasoning!"
        
        memory_context = ""
        if memory_summary:
            memory_context = f"\n🧠 MEMORY: {memory_summary[:200]}..."
        
        # Add linguistic fluidity to basic prompt
        fluidity_instruction = "\nUse clear, relatable, and human language. Avoid sounding robotic or overly academic."
        basic_prompt = f"""BASIC ARGUMENT MODE (Stamina: {stamina})
NET RL SCORE: {rl_status.get('net_score', 0)}{synthesis_urgency}{rl_warning}{memory_context}

{prompt}
{fluidity_instruction}

QUALITY REMINDER: Even basic arguments are evaluated! Poor reasoning = severe penalties!
Learn from your memory context to avoid repeating mistakes.
Provide a direct, well-reasoned argument (2-3 sentences) that shows clear thinking and a human, conversational tone:"""
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": basic_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.85,
                    "num_predict": 300,
                    "stop": ["STAMINA:", "NET:", "USER:", "HUMAN:", "QUALITY:"],
                    "repeat_penalty": 1.1,
                    "num_ctx": 2048
                }
            }
            
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                return self._clean_response(answer)
            else:
                return f"Error: {response.status_code}"
                
        except Exception as e:
            return f"Error: {str(e)[:30]}"
    
    def _get_complexity_guidance(self, complexity_level: str) -> str:
        """Get complexity-appropriate guidance"""
        guidance = {
            "simple": "SIMPLE QUESTION: Provide direct, clear answer. Don't overthink or over-complicate.",
            "moderate": "MODERATE QUESTION: Use solid reasoning but don't over-engineer the analysis.",
            "complex": "COMPLEX QUESTION: Sophisticated analysis required. Use advanced reasoning frameworks.",
            "very_complex": "VERY COMPLEX QUESTION: Deep, multi-layered analysis essential. Apply highest-level reasoning."
        }
        return guidance.get(complexity_level, "")
    
    def _clean_response(self, response: str) -> str:
        """Clean response and ensure completion"""
        if not response:
            return "No response"
        
        # Remove prefixes
        prefixes = ["Assistant:", "AI:", f"{self.side}:", "Response:", "Answer:"]
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Ensure proper sentence completion
        if response and not response.endswith(('.', '!', '?')):
            sentences = response.split('.')
            if len(sentences) > 1:
                complete_sentences = '.'.join(sentences[:-1]) + '.'
                if len(complete_sentences) > 40:
                    response = complete_sentences
                else:
                    response = response.rstrip() + "."
            else:
                response = response.rstrip() + "."
        
        return response

class StrictEnhancedLotusArtRL:
    """Strict Enhanced Lotus Art system with comprehensive RL, quality enforcement, and memory graph"""
    
    def __init__(self, model="llama3.2:3b", rules_file_path="reasoning_rules.json"):
        print("🌸 STRICT ENHANCED LOTUS ART SYSTEM WITH RL, QUALITY ENFORCEMENT & MEMORY")
        print("🧠 NEW: Emotional Awareness, Complexity Analysis & Situational Awareness")
        print("📊 NEW: Graph-based Memory System for Debate History")
        print("🔧 NEW: Modular JSON-based Reasoning Rules System")
        print("=" * 80)
        
        # Initialize components with rules file
        self.rules_file_path = rules_file_path
        self.rag = EnhancedReasoningRulesRAG(rules_file_path)
        self.mentor_evaluator = EnhancedMentorEvaluator()
        self.pro_debater = OllamaDebater(model, "pro")
        self.con_debater = OllamaDebater(model, "con")
        
        # RL components with rule costs from JSON
        self.stamina_manager = EnhancedStaminaManager(18, self.rag.rules_loader)
        self.rl_system = StrictRewardPunishmentSystem()
        
        # Memory graph system
        self.memory_graph = DebateMemoryGraph()
        
        # Tracking
        self.debate_history = []
        self.evaluation_history = []
        
        if self.mentor_evaluator.available:
            print("✅ Strict Enhanced Lotus Art with RL and Memory ready!")
        else:
            print("❌ Ollama not available")
    
    def reload_rules(self):
        """Reload reasoning rules from JSON file"""
        print("🔄 Reloading reasoning rules...")
        self.rag.reload_rules()
        # Update stamina manager with new rule costs
        self.stamina_manager.rule_costs = self.rag.rules_loader.get_rule_costs()
        print("✅ Rules reloaded successfully")
    
    def conduct_strict_rl_debate(self, question: str):
        """Conduct debate with strict RL quality enforcement and memory graph"""
        print(f"\n🥊 STRICT RL QUALITY-ENFORCED DEBATE WITH MEMORY: {question}")
        print("=" * 80)
        
        # Clear any existing history for fresh start
        self.memory_graph.clear_history()
        # Fetch initial internet data for the topic
        self.mentor_evaluator.update_internet_context(question, turn=0)
        
        self.current_topic = question
        turn_count = 0
        consecutive_failures = 0
        max_turns = 18  # Limit debate to 18 rounds
        
        mentor_intervention = ""
        # Add internal reflection context for debaters and mentor
        self.pro_reflection_context = ""
        self.con_reflection_context = ""
        self.mentor_reflection_context = ""
        while turn_count < max_turns:
            turn_count += 1
            print(f"\n🔄 Turn {turn_count}")
            print("-" * 60)
            # Update internet data at each self-reflection turn (every 4 turns)
            if turn_count == 1 or turn_count % 4 == 0:
                self.mentor_evaluator.update_internet_context(question, turn=turn_count)
            
            # Check stamina status and synthesis triggers
            status = self.stamina_manager.get_status()
            synthesis_trigger = status['synthesis_trigger']
            synthesis_level = status['synthesis_level']
            # --- Ensure synthesis mode is properly activated ---
            if synthesis_trigger or status['both_exhausted']:
                synthesis_level = 'emergency' if status['both_exhausted'] else synthesis_level
            
            print(f"💪 Stamina - Pro: {status['pro_stamina']}, Con: {status['con_stamina']}")
            print(f"🎯 RL Performance: {self.rl_system.get_performance_summary()}")
            
            if synthesis_trigger:
                print(f"🔮 SYNTHESIS MODE: {synthesis_level.upper()}")
            
            # Multiple termination conditions
            if status['both_exhausted']:
                print(f"\n🏁 Both sides exhausted! Ending debate.")
                break
            
            if consecutive_failures >= 3:
                print(f"\n🏁 Too many consecutive failures! Ending debate.")
                break
            
            turn_successful = False
            
            # Pro-side turn with RL evaluation and memory
            if not status['pro_exhausted']:
                pro_result = self._execute_strict_rl_turn_with_memory("pro", question, synthesis_level, turn_count, mentor_intervention)
                if pro_result:
                    turn_successful = True
                    self._display_turn_result("pro", pro_result, status['pro_stamina'])
                    # Mentor analyzes and intervenes for next turn
                    mentor_intervention = self.mentor_evaluator.mentor_intervene(
                        question, "pro", pro_result, self.memory_graph, self.rag.reasoning_rules
                    )
                else:
                    mentor_intervention = ""
            
            # Con-side turn with RL evaluation and memory
            if not status['con_exhausted']:
                con_result = self._execute_strict_rl_turn_with_memory("con", question, synthesis_level, turn_count, mentor_intervention)
                if con_result:
                    turn_successful = True
                    self._display_turn_result("con", con_result, status['con_stamina'])
                    mentor_intervention = self.mentor_evaluator.mentor_intervene(
                        question, "con", con_result, self.memory_graph, self.rag.reasoning_rules
                    )
                else:
                    mentor_intervention = ""
            
            # Apply RL consequences after both turns
            self._apply_rl_consequences()
            
            # Track consecutive failures
            if turn_successful:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            
            time.sleep(0.8)
            # After every 4 turns, insert self-reflection (internal only)
            if turn_count % 4 == 0:
                pro_reflection = self.pro_debater.generate_argument_with_stamina(
                    f"SELF-REFLECTION: Reflect on your arguments so far. What has worked? What hasn't? What will you try next?", [], self.stamina_manager.pro_stamina, 'none', {'reward_bank': self.rl_system.get_reward_bank('pro'), 'punishment_debt': self.rl_system.get_punishment_debt('pro'), 'net_score': self.rl_system.get_net_performance('pro')}, self.memory_graph.get_memory_summary('pro'))
                con_reflection = self.con_debater.generate_argument_with_stamina(
                    f"SELF-REFLECTION: Reflect on your arguments so far. What has worked? What hasn't? What will you try next?", [], self.stamina_manager.con_stamina, 'none', {'reward_bank': self.rl_system.get_reward_bank('con'), 'punishment_debt': self.rl_system.get_punishment_debt('con'), 'net_score': self.rl_system.get_net_performance('con')}, self.memory_graph.get_memory_summary('con'))
                mentor_reflection = self.mentor_evaluator.generate_response(
                    f"META-REFLECTION: As the mentor, reflect on the debate so far. What are the main strengths, weaknesses, and what should each side focus on next?", 120)
                # Update internal context for learning
                self.pro_reflection_context = pro_reflection
                self.con_reflection_context = con_reflection
                self.mentor_reflection_context = mentor_reflection
                # Update memory based on reflection
                self.memory_graph.update_from_reflection('pro', pro_reflection)
                self.memory_graph.update_from_reflection('con', con_reflection)
                # Write full details to output.txt
                with open('output.txt', 'a', encoding='utf-8') as f:
                    import json
                    f.write(f"\nSELF-REFLECTION BREAK (Turn {turn_count})\n")
                    f.write(f"PRO: {pro_reflection}\nCON: {con_reflection}\nMENTOR: {mentor_reflection}\n")
            
            # --- Emergent Synthesis: Occasionally inject a creative, brain-inspired frame ---
            if turn_count % 5 == 0 and turn_count > 0:
                emergent = self.rag.emergent_synthesis_frame(question, memory_summary=self.memory_graph.get_memory_summary('pro'))
                print(f"\n🧠🌟 THOUSAND BRAINS EMERGENT SYNTHESIS FRAME:\n{emergent['frame']}")
                # Use this as a creative, parallel-processing context for both sides
                self.pro_reflection_context = emergent['frame']
                self.con_reflection_context = emergent['frame']
        
        return self._conclude_strict_debate_with_memory(question)
    
    def _execute_strict_rl_turn_with_memory(self, side: str, question: str, synthesis_level: str, turn_count: int, mentor_intervention: str = "", reflection_context: str = None, mentor_reflection: str = None) -> Optional[Dict]:
        """Execute a single debate turn with strict RL evaluation and memory integration"""
        
        # Check if side can make any argument
        if not self.stamina_manager.can_afford_basic_argument(side):
            return None
        
        # Get memory summary for this side
        memory_summary = self.memory_graph.get_memory_summary(side)
        memory_insights = self.memory_graph.get_memory_insights(side)
        
        # Get context for prompt, now with mentor intervention
        context = self._build_context_with_memory(question, side, memory_summary, mentor_intervention)
        
        # Get current stamina and RL status
        current_stamina = self.stamina_manager.pro_stamina if side == "pro" else self.stamina_manager.con_stamina
        rl_status = {
            'reward_bank': self.rl_system.get_reward_bank(side),
            'punishment_debt': self.rl_system.get_punishment_debt(side),
            'net_score': self.rl_system.get_net_performance(side)
        }
        # Emergent synthesis: generate columns and synthesize if in emergent frame context
        use_emergent = hasattr(self, f'{side}_reflection_context') and self.__dict__.get(f'{side}_reflection_context', '').startswith('Thousand Brains Emergent Synthesis Frame:')
        if use_emergent:
            # Parse columns from context
            import re
            context_text = self.__dict__.get(f'{side}_reflection_context', '')
            columns = []
            for match in re.finditer(r'\[Column (\d+)\](.*?)Framework:(.*?)\n', context_text, re.DOTALL):
                rule_text = match.group(2).strip()
                framework = match.group(3).strip()
                columns.append({'rule_text': rule_text, 'framework': framework})
            if columns:
                gen = self.rag.generate_columns_and_synthesis(context, columns, side, current_stamina, synthesis_level, rl_status, memory_summary)
                response = gen['final_argument']
                sub_arguments = gen['sub_arguments']
                actual_cost = len(sub_arguments) + 1
                # For emergent, we don't have rule dicts, so use rule_text as a stand-in for rule_id
                rules_used = [c['rule_text'] for c in columns]
                affordable_rules = []  # Ensure defined for downstream calls
                is_fallback = False
            else:
                response = None
                actual_cost = 1
                is_fallback = True
                rules_used = []
                affordable_rules = []
        else:
            # Try to get affordable rules first (now with memory context)
            prefer_synthesis = synthesis_level in ["standard", "critical", "emergency"]
            rules = self.rag.retrieve_relevant_rules(
                context, 5, prefer_synthesis=prefer_synthesis, memory_summary=memory_summary,
                side=side, stamina=current_stamina, memory_graph=self.memory_graph,
                reflection_context=reflection_context, mentor_reflection=mentor_reflection
            )
            affordable_rules, cost = self.stamina_manager.can_afford_rules(side, rules)
            response = None
            actual_cost = 1
            is_fallback = False
            min_length = 10
            max_attempts = 3
            attempts = 0
            while attempts < max_attempts:
                if affordable_rules and cost > 0:
                    debater = self.pro_debater if side == "pro" else self.con_debater
                    response = debater.generate_argument_with_stamina(
                        context, affordable_rules, current_stamina, synthesis_level, rl_status, memory_summary
                    )
                    actual_cost = cost
                elif current_stamina >= 1:
                    debater = self.pro_debater if side == "pro" else self.con_debater
                    response = debater.generate_basic_argument(context, current_stamina, synthesis_level, rl_status, memory_summary)
                    actual_cost = 1
                    is_fallback = True
                    affordable_rules = []
                else:
                    return None
                if response and len(response.strip()) >= min_length:
                    break
                attempts += 1
            if not response or len(response.strip()) < min_length:
                logging.debug(f"Empty/short argument generated for {side} on turn {turn_count}")
                return None
            # Always use rule_id strings for rules_used
            rules_used = [rule['rule_id'] for rule in affordable_rules] if affordable_rules else ["basic_argument"]
        if response and self.stamina_manager.spend_stamina(side, actual_cost):
            evaluation = self.mentor_evaluator.evaluate_argument_quality(
                response, affordable_rules, side, context, memory_summary, self.memory_graph
            )
            rl_result = self.rl_system.evaluate_and_reward(
                side, response, affordable_rules, evaluation, memory_insights
            )
            node_id = self.memory_graph.add_argument(
                side=side,
                argument=response,
                rules_used=rules_used,
                quality_score=evaluation['reasoning_quality'],
                rl_result=rl_result,
                stamina_before=current_stamina,
                stamina_after=current_stamina - actual_cost,
                synthesis_level=synthesis_level,
                frames_used=rules_used if use_emergent and columns else None
            )
            turn_data = {
                side: response,
                f"{side}_stamina_before": current_stamina,
                f"{side}_stamina_after": current_stamina - actual_cost,
                f"{side}_rules_used": rules_used,
                f"{side}_cost": actual_cost,
                "synthesis_level": synthesis_level,
                "fallback_mode": is_fallback,
                "turn": turn_count,
                "evaluation": evaluation,
                "rl_result": rl_result,
                "memory_node_id": node_id,
                "sub_arguments": sub_arguments if use_emergent and columns else None
            }
            self.debate_history.append(turn_data)
            self.evaluation_history.append({
                "side": side,
                "turn": turn_count,
                "evaluation": evaluation,
                "rl_result": rl_result,
                "memory_node_id": node_id
            })
            return {
                "response": response,
                "rules_used": rules_used,
                "cost": actual_cost,
                "fallback": is_fallback,
                "evaluation": evaluation,
                "rl_result": rl_result,
                "memory_node_id": node_id,
                "sub_arguments": sub_arguments if use_emergent and columns else None
            }
        return None
    
    def _display_turn_result(self, side: str, result: Dict, original_stamina: int):
        """Display turn result: ONLY pro/con response and stamina (no RL, no framework, no self-acknowledgement, no meta-analytic/acknowledgement text)"""
        response = result['response']
        new_stamina = original_stamina - result['cost']
        # Remove self-acknowledgement and meta-analytic phrases from response for maturity
        meta_phrases = [
            "argument demonstrates sophisticated reasoning by", "argument demonstrates advanced reasoning by", "argument aims to provide a strong foundation", "By employing these advanced reasoning frameworks", "By using these advanced reasoning frameworks", "By employing these frameworks", "By using these frameworks", "This argument demonstrates", "This argument aims to", "By employing", "By using", "By demonstrating", "By providing", "By integrating", "By acknowledging", "By drawing upon", "By considering", "By recognizing", "By focusing on", "By shifting the focus", "By reducing", "By applying", "By attempting to", "By addressing", "By introducing", "By identifying", "By prioritizing", "By highlighting", "By emphasizing", "By illustrating", "By supporting", "By defending", "By justifying", "By reasoning", "By contending", "By maintaining", "By asserting", "By claiming", "By suggesting", "By proposing", "By recommending", "By encouraging", "By urging", "By advising", "By cautioning", "By warning", "By noting", "By observing", "By remarking", "By commenting", "By pointing out", "By mentioning", "By stating", "By declaring", "By announcing", "By proclaiming", "By pronouncing", "By affirming", "By confirming", "By verifying", "By validating", "By authenticating", "By substantiating", "By corroborating", "By attesting", "By testifying", "By swearing", "By vouching", "By guaranteeing", "By assuring", "By promising", "By pledging", "By vowing", "By avowing", "By professing", "By confessing", "By admitting", "By acknowledging", "By conceding", "By granting", "By allowing", "By accepting", "By recognizing", "By realizing", "By understanding", "By comprehending", "By grasping", "By apprehending", "By perceiving", "By discerning", "By detecting", "By noticing", "By seeing", "By hearing", "By feeling", "By sensing", "By experiencing", "By encountering", "By meeting", "By facing", "By dealing with", "By handling", "By managing", "By coping with", "By addressing", "By tackling", "By approaching", "By considering", "By reflecting", "By thinking", "By pondering", "By contemplating", "By deliberating", "By meditating", "By musing", "By speculating", "By supposing", "By assuming", "By presuming", "By guessing", "By estimating", "By calculating", "By computing", "By figuring", "By working out", "By deducing", "By inferring", "By concluding", "By reasoning", "By surmising", "By conjecturing", "By hypothesizing", "By theorizing", "By postulating"
        ]
        for phrase in meta_phrases:
            response = response.replace(phrase, "")
        # Remove any lines starting with meta-analytic/acknowledgement markers
        import re
        response = re.sub(r"(?im)^.*(argument demonstrates|argument aims to|argument provides|argument supports|argument defends|argument justifies|argument reasons|argument contends|argument maintains|argument asserts|argument claims|argument suggests|argument proposes|argument recommends|argument encourages|argument urges|argument advises|argument cautions|argument warns|argument notes|argument observes|argument remarks|argument comments|argument points out|argument mentions|argument states|argument declares|argument announces|argument proclaims|argument pronounces|argument affirms|argument confirms|argument verifies|argument validates|argument authenticates|argument substantiates|argument corroborates|argument attests|argument testifies|argument swears|argument vouches|argument guarantees|argument assures|argument promises|argument pledges|argument vows|argument avows|argument professes|argument confesses|argument admits|argument acknowledges|argument concedes|argument grants|argument allows|argument accepts|argument recognizes|argument realizes|argument understands|argument comprehends|argument grasps|argument apprehends|argument perceives|argument discerns|argument detects|argument notices|argument sees|argument hears|argument feels|argument senses|argument experiences|argument encounters|argument meets|argument faces|argument deals with|argument handles|argument manages|argument copes with|argument addresses|argument tackles|argument approaches|argument considers|argument reflects|argument thinks|argument ponders|argument contemplates|argument deliberates|argument meditates|argument muses|argument speculates|argument supposes|argument assumes|argument presumes|argument guesses|argument estimates|argument calculates|argument computes|argument figures|argument works out|argument deduces|argument infers|argument concludes|argument reasons|argument surmises|argument conjectures|argument hypothesizes|argument theorizes|argument postulates|argument provides a strong foundation|argument provides a strong basis|argument provides a strong case|argument provides a strong argument|argument provides a strong point|argument provides a strong perspective|argument provides a strong view|argument provides a strong stance|argument provides a strong position|argument provides a strong opinion|argument provides a strong belief|argument provides a strong conviction|argument provides a strong assertion|argument provides a strong claim|argument provides a strong suggestion|argument provides a strong proposal|argument provides a strong recommendation|argument provides a strong encouragement|argument provides a strong urging|argument provides a strong advising|argument provides a strong cautioning|argument provides a strong warning|argument provides a strong noting|argument provides a strong observing|argument provides a strong remarking|argument provides a strong commenting|argument provides a strong pointing out|argument provides a strong mentioning|argument provides a strong stating|argument provides a strong declaring|argument provides a strong announcing|argument provides a strong proclaiming|argument provides a strong pronouncing|argument provides a strong affirming|argument provides a strong confirming|argument provides a strong verifying|argument provides a strong validating|argument provides a strong authenticating|argument provides a strong substantiating|argument provides a strong corroborating|argument provides a strong attesting|argument provides a strong testifying|argument provides a strong swearing|argument provides a strong vouching|argument provides a strong guaranteeing|argument provides a strong assuring|argument provides a strong promising|argument provides a strong pledging|argument provides a strong vowing|argument provides a strong avowing|argument provides a strong professing|argument provides a strong confessing|argument provides a strong admitting|argument provides a strong acknowledging|argument provides a strong conceding|argument provides a strong granting|argument provides a strong allowing|argument provides a strong accepting|argument provides a strong recognizing|argument provides a strong realizing|argument provides a strong understanding|argument provides a strong comprehending|argument provides a strong grasping|argument provides a strong apprehending|argument provides a strong perceiving|argument provides a strong discerning|argument provides a strong detecting|argument provides a strong noticing|argument provides a strong seeing|argument provides a strong hearing|argument provides a strong feeling|argument provides a strong sensing|argument provides a strong experiencing|argument provides a strong encountering|argument provides a strong meeting|argument provides a strong facing|argument provides a strong dealing with|argument provides a strong handling|argument provides a strong managing|argument provides a strong coping with|argument provides a strong addressing|argument provides a strong tackling|argument provides a strong approaching|argument provides a strong considering|argument provides a strong reflecting|argument provides a strong thinking|argument provides a strong pondering|argument provides a strong contemplating|argument provides a strong deliberating|argument provides a strong meditating|argument provides a strong musing|argument provides a strong speculating|argument provides a strong supposing|argument provides a strong assuming|argument provides a strong presuming|argument provides a strong guessing|argument provides a strong estimating|argument provides a strong calculating|argument provides a strong computing|argument provides a strong figuring|argument provides a strong working out|argument provides a strong deducing|argument provides a strong inferring|argument provides a strong concluding|argument provides a strong reasoning|argument provides a strong surmising|argument provides a strong conjecturing|argument provides a strong hypothesizing|argument provides a strong theorizing|argument provides a strong postulating).*\n", "", response)
        # Remove any 'Batches:' lines
        response = re.sub(r"(?im)^Batches:.*\n?", "", response)
        # Add clear markers for pro/con
        marker = f"=== {side.upper()} ==="
        print(f"{marker}\n{response.strip()}\nStamina: {new_stamina}")
    
    def _apply_rl_consequences(self):
        """Apply RL consequences - stamina exchanges and penalties, only print stamina exchange info"""
        for side in ["pro", "con"]:
            # Check for reward exchange opportunities
            if self.rl_system.can_exchange_for_stamina(side):
                stamina_gained = self.rl_system.exchange_rewards_for_stamina(side)
                if stamina_gained > 0:
                    self.stamina_manager.apply_reward_exchange(side, stamina_gained)
                    print(f"{side.upper()} exchanged rewards for +{stamina_gained} stamina!")
            # Apply punishment penalties (do not print)
            stamina_lost = self.rl_system.apply_punishment_debt(side)
            if stamina_lost > 0:
                self.stamina_manager.apply_punishment_loss(side, stamina_lost)
        # No other output
    
    def _build_context_with_memory(self, question: str, side: str, memory_summary: str, mentor_intervention: str = "") -> str:
        """Build context for debate prompt with memory integration and structured insights"""
        base_context = ""
        if self.debate_history:
            last_turn = self.debate_history[-1]
            opponent_side = "con" if side == "pro" else "pro"
            if opponent_side in last_turn:
                opponent_response = last_turn[opponent_side]
                opponent_quality = last_turn.get('evaluation', {}).get('reasoning_quality', 5.0)
                base_context = f"{question}\nOpponent's argument (Quality: {opponent_quality:.1f}/10): {opponent_response}\nRespond with superior reasoning:"
            else:
                base_context = f"{question}\nProvide high-quality opening argument with sophisticated reasoning:"
        else:
            base_context = f"{question}\nProvide exceptional opening argument that demonstrates advanced reasoning:"
        # Add memory context if available
        if memory_summary:
            base_context += f"\n\n[MEMORY CONTEXT: {memory_summary}]"
        # Add structured memory insights
        memory_insights = self.memory_graph.get_memory_insights(side)
        if memory_insights:
            insight_lines = ["\nMEMORY INSIGHTS:"]
            if memory_insights.get('top_own_arguments'):
                insight_lines.append("- Your best arguments: " + "; ".join(memory_insights['top_own_arguments']))
            if memory_insights.get('top_opp_arguments'):
                insight_lines.append("- Opponent's best arguments: " + "; ".join(memory_insights['top_opp_arguments']))
            if memory_insights.get('best_rules'):
                best = memory_insights['best_rules'][0] if memory_insights['best_rules'] else None
                if best:
                    insight_lines.append(f"- Most effective rule: {best[0]} (avg {best[1]['avg_quality']:.1f})")
            if memory_insights.get('worst_rules'):
                worst = memory_insights['worst_rules'][0] if memory_insights['worst_rules'] else None
                if worst:
                    insight_lines.append(f"- Least effective rule: {worst[0]} (avg {worst[1]['avg_quality']:.1f})")
            if memory_insights.get('repeated_arguments'):
                insight_lines.append("- Avoid repeating: " + "; ".join(memory_insights['repeated_arguments']))
            if memory_insights.get('quality_trend'):
                trend = memory_insights['quality_trend']
                insight_lines.append(f"- Quality trend: {trend['trend']} (recent avg: {trend.get('recent_quality', 0):.1f})")
        # Add latest reflection context for learning (internal only)
        if hasattr(self, 'pro_reflection_context') and side == 'pro' and self.pro_reflection_context:
            base_context += f"\n\n[SELF-REFLECTION: {self.pro_reflection_context}]"
        if hasattr(self, 'con_reflection_context') and side == 'con' and self.con_reflection_context:
            base_context += f"\n\n[SELF-REFLECTION: {self.con_reflection_context}]"
        if hasattr(self, 'mentor_reflection_context') and self.mentor_reflection_context:
            base_context += f"\n\n[MENTOR META-REFLECTION: {self.mentor_reflection_context}]"
        if mentor_intervention:
            base_context += f"\n\n[MENTOR INTERVENTION: {mentor_intervention}]"
        # Add mentor internet data context
        if hasattr(self, 'mentor_evaluator') and hasattr(self.mentor_evaluator, 'format_internet_context_for_debators'):
            internet_context = self.mentor_evaluator.format_internet_context_for_debators(question)
            if internet_context:
                base_context += f"\n\n{internet_context}"
        return base_context
    
    def _conclude_strict_debate_with_memory(self, question: str) -> Dict:
        """Conclude debate with comprehensive RL performance analysis and memory insights"""
        print(f"\n🏁 STRICT DEBATE CONCLUSION & MEMORY ANALYSIS")
        print("=" * 70)
        
        # Determine exhaustion reason
        status = self.stamina_manager.get_status()
        
        if status['both_exhausted']:
            exhaustion_reason = "Both sides exhausted their reasoning stamina"
        elif status['pro_exhausted']:
            exhaustion_reason = "Pro-side exhausted reasoning stamina"
        elif status['con_exhausted']:
            exhaustion_reason = "Con-side exhausted reasoning stamina"
        else:
            exhaustion_reason = "Debate ended due to turn limit or failure threshold"
        
        print(f"📊 Final Status: {exhaustion_reason}")
        print(f"💪 Final Stamina - Pro: {status['pro_stamina']}, Con: {status['con_stamina']}")
        
        # Extract arguments from history
        pro_arguments = []
        con_arguments = []
        
        for turn in self.debate_history:
            if 'pro' in turn:
                pro_arguments.append(turn['pro'])
            if 'con' in turn:
                con_arguments.append(turn['con'])
        
        print(f"📈 Total Arguments - Pro: {len(pro_arguments)}, Con: {len(con_arguments)}")
        
        # Get comprehensive performance summary
        performance_summary = self.rl_system.get_performance_summary()
        
        # Display RL Performance Analysis
        print(f"\n🎯 RL PERFORMANCE ANALYSIS")
        print("-" * 40)
        for side in ["pro", "con"]:
            perf = performance_summary[side]
            print(f"{side.title()}: Net Score={perf['net_score']}, Rewards={perf['reward_bank']}, Debt={perf['punishment_debt']}, Evaluations={perf['evaluation_count']}")
        
        # Memory Graph Analysis
        self._analyze_memory_insights()
        
        # Analyze special rule usage including situational awareness
        self._analyze_special_rule_usage_with_memory()
        
        # Generate mentor synthesis with performance context and memory insights
        final_synthesis = "No substantial debate occurred"
        mentor_error = None
        polished_answer = None
        if pro_arguments or con_arguments:
            print(f"\n🎓 MENTOR SYNTHESIS WITH PERFORMANCE & MEMORY CONTEXT")
            print("-" * 50)
            try:
                final_synthesis = self.mentor_evaluator.synthesize_final_answer(
                    question, pro_arguments, con_arguments, exhaustion_reason, performance_summary, self.memory_graph
                )
                print(f"🔮 FINAL SYNTHESIS: {final_synthesis}")
                polished_answer = self.mentor_evaluator.generate_response(
                    f"You are a world-class reasoning mentor. Based on the entire debate, provide a comprehensive, detailed, and polished final answer to the following question, as a high-parameter model would.\n"
                    f"- Integrate all novel approaches, emergent insights, and hypotheses developed during the debate.\n"
                    f"- Explicitly reference creative leaps, new analogies, and unique synthesis points that arose.\n"
                    f"- Make the answer structured, clear, and as insightful as possible.\n"
                    f"- Reference the debate's most original contributions and emergent ideas.\n"
                    f"QUESTION: {question}\n\nDebate context:\nPro: {' | '.join(pro_arguments[-2:])}\nCon: {' | '.join(con_arguments[-2:])}\n\n[You may use the following as inspiration for novel synthesis:]\n{final_synthesis}",
                    max_tokens=600
                )
            except Exception as e:
                mentor_error = str(e)
                polished_answer = f"❌ Mentor synthesis failed: {mentor_error}"
                print(polished_answer)
            # Print and log the mentor answer or error
            if polished_answer is None or not polished_answer.strip():
                print("❌ Mentor final answer is missing or empty!")
                polished_answer = "❌ Mentor final answer is missing or empty!"
            elif any(x in polished_answer.lower() for x in ["error", "unavailable", "failed"]):
                print(f"❌ Mentor final answer error: {polished_answer}")
            else:
                print(f"\n=== MENTOR FINAL ANSWER ===\n{polished_answer}\n")
            # Always write mentor answer to file
            try:
                with open('mentor_final_answer.txt', 'w', encoding='utf-8') as f:
                    f.write(polished_answer.strip() + '\n')
            except Exception as e:
                print(f"❌ Failed to write mentor_final_answer.txt: {e}")
        
        # Calculate quality metrics
        pro_quality_scores = [eval_data['evaluation']['reasoning_quality'] for eval_data in self.evaluation_history if eval_data['side'] == 'pro']
        con_quality_scores = [eval_data['evaluation']['reasoning_quality'] for eval_data in self.evaluation_history if eval_data['side'] == 'con']
        
        pro_avg_quality = sum(pro_quality_scores) / len(pro_quality_scores) if pro_quality_scores else 0
        con_avg_quality = sum(con_quality_scores) / len(con_quality_scores) if con_quality_scores else 0
        
        # Calculate reasoning costs
        total_pro_cost = sum(turn.get('pro_cost', 0) for turn in self.debate_history)
        total_con_cost = sum(turn.get('con_cost', 0) for turn in self.debate_history)
        
        # Memory analysis results
        memory_analysis = self._get_memory_analysis_summary()
        
        results = {
            'question': question,
            'exhaustion_reason': exhaustion_reason,
            'final_synthesis': final_synthesis,
            'total_turns': len(self.debate_history),
            'final_stamina': status,
            'rl_performance': performance_summary,
            'quality_metrics': {
                'pro_average_quality': pro_avg_quality,
                'con_average_quality': con_avg_quality,
                'pro_quality_scores': pro_quality_scores,
                'con_quality_scores': con_quality_scores
            },
            'reasoning_costs': {
                'pro_total_cost': total_pro_cost,
                'con_total_cost': total_con_cost,
                'pro_average_cost': total_pro_cost / max(len(pro_arguments), 1),
                'con_average_cost': total_con_cost / max(len(con_arguments), 1)
            },
            'memory_analysis': memory_analysis,
            'debate_history': self.debate_history,
            'evaluation_history': self.evaluation_history
        }
        
        # Display comprehensive results
        print(f"\n📊 COMPREHENSIVE PERFORMANCE REPORT WITH MEMORY")
        print("=" * 50)
        print(f"🎯 Question: {question}")
        print(f"🔮 Synthesis: {final_synthesis}")
        print(f"⚡ Exhaustion: {exhaustion_reason}")
        print(f"🔄 Total Turns: {len(self.debate_history)}")
        print(f"🏆 Quality Performance:")
        print(f"   Pro: Avg Quality={pro_avg_quality:.2f}/10, Net RL={performance_summary['pro']['net_score']}")
        print(f"   Con: Avg Quality={con_avg_quality:.2f}/10, Net RL={performance_summary['con']['net_score']}")
        print(f"💰 Final RL Status:")
        print(f"   Pro: {performance_summary['pro']['reward_bank']} rewards, {performance_summary['pro']['punishment_debt']} debt")
        print(f"   Con: {performance_summary['con']['reward_bank']} rewards, {performance_summary['con']['punishment_debt']} debt")
        print(f"🧠 Reasoning Efficiency:")
        print(f"   Pro: Cost={total_pro_cost}, Avg={total_pro_cost/max(len(pro_arguments), 1):.1f}")
        print(f"   Con: Cost={total_con_cost}, Avg={total_con_cost/max(len(con_arguments), 1):.1f}")
        print(f"📊 Memory Performance:")
        print(f"   Pro trend: {memory_analysis['pro_trend']}, Con trend: {memory_analysis['con_trend']}")
        print(f"   Total memory nodes: {memory_analysis['total_nodes']}")
        
        return results
    
    def _analyze_memory_insights(self):
        """Analyze memory graph insights"""
        print(f"\n🧠📊 MEMORY GRAPH ANALYSIS")
        print("-" * 40)
        
        # Get quality trends
        pro_trends = self.memory_graph.get_quality_trends("pro")
        con_trends = self.memory_graph.get_quality_trends("con")
        
        print(f"Quality Evolution:")
        print(f"   Pro: {pro_trends['trend']} (recent: {pro_trends.get('recent_quality', 0):.1f}, earlier: {pro_trends.get('earlier_quality', 0):.1f})")
        print(f"   Con: {con_trends['trend']} (recent: {con_trends.get('recent_quality', 0):.1f}, earlier: {con_trends.get('earlier_quality', 0):.1f})")
        
        # Get rule effectiveness
        pro_rules = self.memory_graph.get_rule_effectiveness("pro")
        con_rules = self.memory_graph.get_rule_effectiveness("con")
        
        print(f"\nMost Effective Rules:")
        if pro_rules['best_rules']:
            best_pro = pro_rules['best_rules'][0]
            print(f"   Pro: {best_pro[0]} (avg: {best_pro[1]['avg_quality']:.1f}, used: {best_pro[1]['usage_count']}x)")
        
        if con_rules['best_rules']:
            best_con = con_rules['best_rules'][0]
            print(f"   Con: {best_con[0]} (avg: {best_con[1]['avg_quality']:.1f}, used: {best_con[1]['usage_count']}x)")
        
        # Conversation flow insights
        flow = self.memory_graph.get_conversation_flow(15)
        if len(flow) >= 4:
            recent_quality = [turn['quality_score'] for turn in flow[-4:]]
            early_quality = [turn['quality_score'] for turn in flow[:4]]
            
            recent_avg = sum(recent_quality) / len(recent_quality)
            early_avg = sum(early_quality) / len(early_quality)
            
            print(f"\nConversation Quality Evolution:")
            print(f"   Early debate avg: {early_avg:.1f}/10")
            print(f"   Recent debate avg: {recent_avg:.1f}/10")
            print(f"   Overall trend: {'Improving' if recent_avg > early_avg else 'Declining' if recent_avg < early_avg else 'Stable'}")
    
    def _analyze_special_rule_usage_with_memory(self):
        """Analyze usage of special rules including situational awareness"""
        print(f"\n🧠💝📊 SPECIAL RULE USAGE ANALYSIS (Including Situational Awareness)")
        print("-" * 60)
        
        complexity_usage = {"pro": 0, "con": 0}
        emotional_usage = {"pro": 0, "con": 0}
        situational_usage = {"pro": 0, "con": 0}
        
        for turn in self.debate_history:
            for side in ["pro", "con"]:
                if f"{side}_rules_used" in turn:
                    rules_used = turn[f"{side}_rules_used"]
                    if "complexity_awareness" in rules_used:
                        complexity_usage[side] += 1
                    if "emotional_awareness" in rules_used:
                        emotional_usage[side] += 1
                    if "situational_awareness" in rules_used:
                        situational_usage[side] += 1
        
        print(f"🧠 Complexity Awareness Usage:")
        print(f"   Pro: {complexity_usage['pro']} times, Con: {complexity_usage['con']} times")
        print(f"💝 Emotional Awareness Usage:")
        print(f"   Pro: {emotional_usage['pro']} times, Con: {emotional_usage['con']} times")
        print(f"📊 Situational Awareness Usage:")
        print(f"   Pro: {situational_usage['pro']} times, Con: {situational_usage['con']} times")
        
        # Analyze effectiveness of each rule
        for rule_type, usage_data, rule_name in [
            ("complexity", complexity_usage, "complexity_awareness"),
            ("emotional", emotional_usage, "emotional_awareness"),
            ("situational", situational_usage, "situational_awareness")
        ]:
            print(f"\n{rule_type.title()} Rule Effectiveness:")
            for side in ["pro", "con"]:
                if usage_data[side] > 0:
                    # Find turns where this rule was used
                    rule_turns = []
                    for turn in self.debate_history:
                        if f"{side}_rules_used" in turn and rule_name in turn[f"{side}_rules_used"]:
                            if 'evaluation' in turn:
                                rule_turns.append(turn['evaluation']['reasoning_quality'])
                    
                    if rule_turns:
                        avg_quality = sum(rule_turns) / len(rule_turns)
                        print(f"   {side.title()}: Avg quality when using {rule_type} rule: {avg_quality:.2f}/10")
                        
                        # Special analysis for situational awareness
                        if rule_name == "situational_awareness":
                            print(f"     → Memory integration effectiveness: {avg_quality:.1f}/10")
    
    def _get_memory_analysis_summary(self) -> Dict:
        """Get summary of memory analysis for results"""
        
        pro_trends = self.memory_graph.get_quality_trends("pro")
        con_trends = self.memory_graph.get_quality_trends("con")
        
        # Count total nodes in memory graph
        total_nodes = len(self.memory_graph.graph.nodes())
        
        # Get conversation flow
        flow = self.memory_graph.get_conversation_flow()
        
        return {
            "pro_trend": pro_trends.get('trend', 'unknown'),
            "con_trend": con_trends.get('trend', 'unknown'),
            "pro_recent_quality": pro_trends.get('recent_quality', 0),
            "con_recent_quality": con_trends.get('recent_quality', 0),
            "total_nodes": total_nodes,
            "conversation_length": len(flow),
            "pro_rule_effectiveness": self.memory_graph.get_rule_effectiveness("pro"),
            "con_rule_effectiveness": self.memory_graph.get_rule_effectiveness("con")
        }

def main():
    """Main function to run the strict RL system with memory and JSON rules"""
    
    try:
        # Test Ollama connection
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama not available!")
            print("Please run: ollama serve")
            print("Then: ollama pull llama3.2:3b")
            return
        
        # Initialize strict RL system with memory and JSON rules
        rules_file = "reasoning_rules.json"
        system = StrictEnhancedLotusArtRL("llama3.2:3b", rules_file)
        
        if not system.mentor_evaluator.available:
            print("⚠️ Mentor evaluator not available")
            return
        
        print(f"\n🔧 Enter your own question to debate a custom topic!")
        custom_question = input("Enter your custom debate question: ").strip()
        if custom_question:
            print(f"\n{'='*90}")
            print(f"🎯 STRICT RL DEBATE WITH MEMORY & JSON RULES (CUSTOM QUESTION)")
            result = system.conduct_strict_rl_debate(custom_question)
            # Print only pro/con responses, stamina, and reward points
            for turn in result['debate_history']:
                pro = turn.get('pro')
                con = turn.get('con')
                pro_stamina = turn.get('pro_stamina_after', turn.get('pro_stamina_before', ''))
                con_stamina = turn.get('con_stamina_after', turn.get('con_stamina_before', ''))
                pro_reward = turn.get('rl_result', {}).get('reward_bank', '') if turn.get('rl_result') and turn.get('side', '') == 'pro' else ''
                con_reward = turn.get('rl_result', {}).get('reward_bank', '') if turn.get('rl_result') and turn.get('side', '') == 'con' else ''
                if pro:
                    print(f"PRO: {pro}\nStamina: {pro_stamina} | Reward Points: {pro_reward}")
                if con:
                    print(f"CON: {con}\nStamina: {con_stamina} | Reward Points: {con_reward}")
            # Write all other details to output.txt
            with open('output.txt', 'w', encoding='utf-8') as f:
                import json
                json.dump(result, f, indent=2)
            print("[Full details written to output.txt]")
        else:
            print("No custom question entered.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure Ollama is running: ollama serve")
        print(f"Also ensure '{rules_file}' exists or run with 'reload' to create it")

if __name__ == "__main__":
    main()