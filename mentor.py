import requests
import logging
from typing import List, Dict
from datetime import datetime
import json

# For web search (you can replace this with a real API or use the provided web_search tool)
def web_search(query, max_results=5):
    """
    Search the web for the given query and return a list of results.
    Each result should be a dict with 'title', 'snippet', and 'url'.
    Uses Brave Search API if available, otherwise returns a static example.
    """
    api_key = 'BSAATsUkVy2eQDFwRh05NeyeyartdIA'  # Brave API key
    endpoint = 'https://api.search.brave.com/res/v1/web/search'
    headers = {
        'Accept': 'application/json',
        'X-Subscription-Token': api_key
    }
    params = {
        'q': query,
        'count': max_results
    }
    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=10)
        results = []
        if response.status_code == 200:
            data = response.json()
            for item in data.get('web', {}).get('results', []):
                results.append({
                    'title': item.get('title', ''),
                    'snippet': item.get('description', ''),
                    'url': item.get('url', '')
                })
        if results:
            return results[:max_results]
    except Exception as e:
        pass  # Fall back to static example if API fails
    # Static fallback for testing
    return [
        {
            'title': 'Example Research Paper on AGI',
            'snippet': 'This paper discusses the feasibility and challenges of artificial general intelligence in the near future.',
            'url': 'https://arxiv.org/abs/1234.5678'
        },
        {
            'title': 'Recent Metrics in AI Progress',
            'snippet': 'A summary of recent benchmarks and metrics in AI research, including human-level performance in various domains.',
            'url': 'https://www.example.com/ai-metrics-summary'
        }
    ][:max_results]

class EnhancedMentorEvaluator:
    """Enhanced mentor with argument quality evaluation capabilities, memory access, and internet data integration"""
    def __init__(self, model_name="llama3.2:3b"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            self.available = response.status_code == 200
            if self.available:
                print("✅ Enhanced Mentor Evaluator connected")
            else:
                print("❌ Ollama not responding")
        except Exception as e:
            print(f"❌ Failed to connect to Ollama: {e}")
            self.available = False
        # Internet data cache for current debate
        self.internet_context_cache = {}

    def fetch_and_filter_internet_data(self, topic: str, max_results: int = 8) -> List[Dict]:
        """Search the web for research, metrics, and data points on the topic, filter for context relevance."""
        # Use web_search to get raw results
        raw_results = web_search(topic, max_results=max_results)
        # Filter and summarize for context
        filtered = []
        for r in raw_results:
            # Simple filter: must mention key topic terms, not be ads, etc.
            if topic.lower() in (r.get('title','').lower() + r.get('snippet','').lower()):
                filtered.append({
                    'title': r.get('title',''),
                    'snippet': r.get('snippet',''),
                    'url': r.get('url','')
                })
        # Optionally, summarize or deduplicate
        return filtered[:max_results]

    def update_internet_context(self, topic: str, turn: int = 0):
        """Update or expand the internet context cache for the topic."""
        if topic not in self.internet_context_cache or turn == 0:
            # Initial fetch
            data = self.fetch_and_filter_internet_data(topic, max_results=8)
            self.internet_context_cache[topic] = data
        else:
            # Progressive update: fetch more or new data
            new_data = self.fetch_and_filter_internet_data(topic, max_results=3)
            # Merge, deduplicate by URL
            urls = {d['url'] for d in self.internet_context_cache[topic]}
            for d in new_data:
                if d['url'] not in urls:
                    self.internet_context_cache[topic].append(d)
                    urls.add(d['url'])

    def get_internet_context(self, topic: str) -> List[Dict]:
        """Return the current filtered internet data for the topic."""
        return self.internet_context_cache.get(topic, [])

    def format_internet_context_for_debators(self, topic: str) -> str:
        """Format the internet data as a context block for debators."""
        data = self.get_internet_context(topic)
        if not data:
            return ""
        lines = ["\n[MENTOR INTERNET DATA CONTEXT]"]
        for d in data:
            lines.append(f"- {d['title']}\n  {d['snippet']}\n  Source: {d['url']}")
        return "\n".join(lines)

    def evaluate_argument_quality(self, argument: str, rules_used: List[Dict], side: str, context: str, memory_summary: str = "", memory_graph=None) -> Dict:
        if not self.available:
            return {"reasoning_quality": 5.0, "synthesis_quality": 0, "feedback": "Evaluator unavailable"}
        frames = ["ethical", "empirical", "pragmatic", "emotional"]
        frame_evaluations = []
        for frame in frames:
            frame_prompt = f"""
            Evaluate the following argument from the {frame} frame (column).
            ARGUMENT: {argument}
            CONTEXT: {context}
            MEMORY: {memory_summary}
            Provide a score (0-10) and a brief critique.
            """
            payload = {
                "model": self.model_name,
                "prompt": frame_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.7,
                    "num_predict": 120,
                    "stop": ["CONTEXT:", "ARGUMENT:", "USER:", "HUMAN:"],
                    "repeat_penalty": 1.1,
                    "num_ctx": 2048
                }
            }
            try:
                response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    evaluation_text = result.get("response", "").strip()
                    frame_evaluations.append(self._parse_evaluation(evaluation_text))
                else:
                    frame_evaluations.append({"reasoning_quality": 5.0, "feedback": f"Error: {response.status_code}"})
            except Exception as e:
                frame_evaluations.append({"reasoning_quality": 5.0, "feedback": f"Error: {str(e)[:30]}"})
        # Aggregate
        avg_quality = sum([f.get("reasoning_quality", 5.0) for f in frame_evaluations]) / len(frame_evaluations)
        feedbacks = [f"[{frames[i]}] {frame_evaluations[i].get('feedback','')}" for i in range(len(frames))]
        # Meta-reasoning/self-critique
        meta_prompt = f"Meta-critique: What might be missing from the above argument? What would a critic or another column say? What are the blind spots or untested assumptions?"
        meta_payload = {
            "model": self.model_name,
            "prompt": meta_prompt + f"\nARGUMENT: {argument}\nCONTEXT: {context}\nMEMORY: {memory_summary}",
            "stream": False,
            "options": {"temperature": 0.4, "top_p": 0.7, "num_predict": 60, "num_ctx": 1024}
        }
        try:
            meta_response = requests.post(f"{self.base_url}/api/generate", json=meta_payload, timeout=15)
            meta_text = meta_response.json().get("response", "").strip() if meta_response.status_code == 200 else ""
        except Exception:
            meta_text = ""
        # Memory analytics
        memory_analytics = ""
        if memory_graph:
            analytics = memory_graph.get_frame_analytics()
            if analytics:
                best = analytics[0]
                underused = [a for a in analytics if a['count'] < 2 and a['avg_quality'] > 7.0]
                memory_analytics = f"Best frame so far: {best['frame']} (avg {best['avg_quality']:.1f}). "
                if underused:
                    memory_analytics += f"Try using underused but effective frame: {underused[0]['frame']}. "
        # Compose feedback
        feedback = " | ".join(feedbacks)
        if memory_analytics:
            feedback += " | " + memory_analytics
        if meta_text:
            feedback += " | Meta-critique: " + meta_text
        return {"reasoning_quality": avg_quality, "synthesis_quality": 0, "feedback": feedback}

    def _parse_evaluation(self, evaluation_text: str) -> Dict:
        scores = {
            "reasoning_quality": 6.0,
            "synthesis_quality": 2.0,
            "clarity": 6.0,
            "feedback": "Evaluation parsing incomplete"
        }
        try:
            lines = evaluation_text.split('\n')
            feedback_found = False
            for line in lines:
                line = line.strip()
                if line.startswith("REASONING_QUALITY:"):
                    try:
                        score_text = line.split(":")[1].strip()
                        score_text = score_text.replace("[", "").replace("]", "").replace("/10", "")
                        scores["reasoning_quality"] = min(10.0, max(0.0, float(score_text)))
                    except (ValueError, IndexError):
                        continue
                elif line.startswith("SYNTHESIS_QUALITY:"):
                    try:
                        score_text = line.split(":")[1].strip()
                        score_text = score_text.replace("[", "").replace("]", "").replace("/10", "")
                        scores["synthesis_quality"] = min(10.0, max(0.0, float(score_text)))
                    except (ValueError, IndexError):
                        continue
                elif line.startswith("CLARITY:"):
                    try:
                        score_text = line.split(":")[1].strip()
                        score_text = score_text.replace("[", "").replace("]", "").replace("/10", "")
                        scores["clarity"] = min(10.0, max(0.0, float(score_text)))
                    except (ValueError, IndexError):
                        continue
                elif line.startswith("FEEDBACK:"):
                    feedback_text = line.split(":", 1)[1].strip()
                    if feedback_text:
                        scores["feedback"] = feedback_text
                        feedback_found = True
            if not feedback_found and evaluation_text:
                import re
                numbers = re.findall(r'(\d+\.?\d*)\s*(?:/10|out of 10)', evaluation_text)
                if numbers:
                    try:
                        scores["reasoning_quality"] = min(10.0, max(0.0, float(numbers[0])))
                    except (ValueError, IndexError):
                        pass
                if len(evaluation_text.strip()) > 10:
                    scores["feedback"] = evaluation_text.strip()[:200]
        except Exception as e:
            scores["feedback"] = f"Parse error: {str(e)[:30]} - content: {evaluation_text[:50]}"
        return scores

    def synthesize_final_answer(self, question: str, pro_arguments: List[str], con_arguments: List[str], exhaustion_reason: str, performance_summary: Dict, memory_graph) -> str:
        import json
        try:
            with open('reasoning_rules.json', 'r', encoding='utf-8') as f:
                rules_data = json.load(f)
            scaffolding = rules_data.get('cognitive_scaffolding_rules', {})
        except Exception:
            scaffolding = {}
        # Decompose question
        decomposition_rule = scaffolding.get('argument_decomposition', {}).get('framework', '')
        decomposition = self.generate_response(f"Decompose the question: {question}\n{decomposition_rule}", 100)
        # Parallel sub-answers (columns)
        frames = ["ethical", "empirical", "pragmatic", "emotional"]
        frame_sections = []
        for frame in frames:
            frame_section = f"[{frame.title()} Column] "
            frame_section += self.generate_response(f"Frame: {frame}\nQuestion: {question}\nPro: {' | '.join(pro_arguments[-2:])}\nCon: {' | '.join(con_arguments[-2:])}\nUse the reasoning rules for {frame} analysis.", 80)
            frame_sections.append(frame_section)
        # Synthesis
        main_synthesis = self.generate_response(f"SYNTHESIS TASK: Integrate the best insights from each column (frame) into a single, creative, novel synthesis.\n{chr(10).join(frame_sections)}", 200)
        # Opponent modeling, meta-reflection as before
        opponent_modeling_rule = scaffolding.get('opponent_modeling', {}).get('framework', '')
        opponent_model = self.generate_response(f"Model the strongest counterargument to the main synthesis.\n{opponent_modeling_rule}\nPro: {' | '.join(pro_arguments[-2:])}\nCon: {' | '.join(con_arguments[-2:])}", 100)
        meta_rule = scaffolding.get('meta_reasoning_self_critique', {}).get('framework', '')
        meta_reflection = self.generate_response(f"After synthesizing, reflect: What might be missing? What would a critic say? {meta_rule}", 100)
        full_output = f"DEBATE SYNTHESIS (Thousand Brains Columns)\n=================\nQUESTION: {question}\n\nDECOMPOSITION:\n{decomposition}\n\nCOLUMNS (Parallel Frames):\n" + "\n\n".join(frame_sections) + f"\n\nMAIN SYNTHESIS:\n{main_synthesis}\n\nOPPONENT MODELING (Strongest Counterargument):\n{opponent_model}\n\nMETA-REFLECTION/SELF-CRITIQUE:\n{meta_reflection}\n\nEXHAUSTION REASON: {exhaustion_reason}\n\nPERFORMANCE SUMMARY: {performance_summary}\n"
        return full_output

    def generate_response(self, prompt: str, max_tokens: int = 300) -> str:
        if not self.available:
            return "Mentor unavailable"
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": max_tokens * 3,
                    "stop": ["Human:", "User:", "Assistant:", "\n\nHuman:", "\n\nUser:", "Round", "ROLE:"],
                    "repeat_penalty": 1.15,
                    "num_ctx": 4096
                }
            }
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "").strip()
                return self._clean_response(response_text, max_tokens)
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)[:50]}"

    def _clean_response(self, response: str, max_tokens: int = 300) -> str:
        if not response:
            return "No response"
        prefixes = ["Assistant:", "AI:", "Mentor:", "Response:", "Answer:"]
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        if response and not response.endswith(('.', '!', '?')):
            sentences = response.split('.')
            if len(sentences) > 1:
                complete_part = '.'.join(sentences[:-1]) + '.'
                if len(complete_part) > 30:
                    response = complete_part
                else:
                    response = response.rstrip() + "."
            else:
                response = response.rstrip() + "."
        return response

    def mentor_intervene(self, question: str, side: str, last_turn_result: dict, memory_graph, reasoning_rules: dict) -> str:
        memory_insights = memory_graph.get_memory_insights(side)
        intervention = ""
        frame_counts = {}
        for turn in memory_graph.get_side_history(side, 20):
            for rule in turn['rules_used']:
                if 'frame' in rule:
                    frame_counts[rule] = frame_counts.get(rule, 0) + 1
        # Use frame analytics for adaptive boosting
        analytics = memory_graph.get_frame_analytics()
        if analytics:
            underused = [a for a in analytics if a['count'] < 2 and a['avg_quality'] > 7.0]
            if underused:
                intervention += f"Try using the underused but effective column/frame '{underused[0]['frame']}'. "
            best = analytics[0]
            intervention += f"Best performing column/frame so far: {best['frame']} (avg {best['avg_quality']:.1f}). "
        if len(frame_counts) < 2:
            intervention += "Try answering from a different reference column/frame (ethical, empirical, pragmatic, emotional). "
        if last_turn_result['response'] in memory_insights.get('repeated_arguments', []):
            intervention += "Avoid repeating previous arguments. "
        best_rules = [r[0] for r in memory_insights.get('best_rules', [])]
        used_rules = last_turn_result.get('rules_used', [])
        for rule in best_rules:
            if rule not in used_rules:
                intervention += f"Consider applying the '{rule}' reasoning rule. "
                break
        trend = memory_insights.get('quality_trend', {})
        if trend.get('trend') == 'declining':
            intervention += "Your recent arguments are declining in quality. Try deeper analysis or a new approach. "
        # Coaching mode: step-by-step guidance
        intervention += "Step-by-step: 1) Try a new column/frame. 2) Integrate memory insights. 3) Synthesize across columns. 4) Self-critique. "
        # Challenge
        import random
        if random.random() < 0.15:
            all_rules = list(reasoning_rules.keys())
            wild_card = random.choice(all_rules)
            intervention += f"Try a wild card rule: '{wild_card}'. "
        rule_usage = {rule: 0 for rule in reasoning_rules}
        for turn in memory_graph.get_side_history(side, 20):
            for rule in turn['rules_used']:
                rule_usage[rule] = rule_usage.get(rule, 0) + 1
        least_used = min(rule_usage, key=rule_usage.get)
        if rule_usage[least_used] == 0:
            intervention += f"Try using the underexplored rule '{least_used}'. "
        import logging
        logging.debug(f"Mentor intervention: {intervention}")
        return intervention.strip()
    # Mentor self-reflection after debate
    def mentor_self_reflection(self, debate_history, memory_graph):
        analytics = memory_graph.get_frame_analytics()
        best = analytics[0] if analytics else None
        worst = analytics[-1] if analytics else None
        reflection = "Mentor Self-Reflection:\n"
        if best:
            reflection += f"Most effective column/frame: {best['frame']} (avg {best['avg_quality']:.1f}).\n"
        if worst:
            reflection += f"Least effective column/frame: {worst['frame']} (avg {worst['avg_quality']:.1f}).\n"
        reflection += "In future debates, I will encourage more use of underused but effective columns/frames, and prompt for more synthesis and meta-critique.\n"
        return reflection 