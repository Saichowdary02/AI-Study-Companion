from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict, Any
import os
import sys
from datetime import datetime

# Check for OpenAI API key - will be loaded from environment
def get_openai_api_key():
    """Get OpenAI API key from environment variables"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("#") or len(api_key.strip()) == 0:
        return None
    return api_key.strip()

# Get API key
api_key = get_openai_api_key()
if not api_key:
    print("Warning: No valid OpenAI API key found. Using fallback mechanisms.")
else:
    print(f"✅ OpenAI API key loaded successfully (length: {len(api_key)})")
# Always try to use real agents - API errors will be handled at runtime

class NoteSummarizerAgent:
    def __init__(self):
        try:
            # Use GPT-3.5-turbo for faster responses, fallback to GPT-4 if needed
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                api_key=api_key,
                temperature=0.1,  # Lower temperature for more consistent summaries
                max_tokens=1500  # Reduced for faster responses
            )
        except Exception as e:
            try:
                print(f"GPT-3.5-turbo not available, trying GPT-4: {e}")
                self.llm = ChatOpenAI(
                    model="gpt-4",
                    api_key=api_key,
                    temperature=0.1,
                    max_tokens=1500
                )
            except Exception as e2:
                print(f"Warning: Failed to initialize OpenAI client for summarization: {e2}")
                self.llm = None

    def summarize_notes(self, text: str) -> str:
        """Summarize lecture notes using OpenAI LLM"""
        if self.llm is None:
            # Fallback: simple text summarization
            return self._generate_fallback_summary(text)

        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Create a well-structured, concise summary of lecture notes for student study. Focus on:

**Key Elements to Include:**
- Main topic/title
- Core concepts and definitions
- Important formulas/theories
- Key examples
- Study tips

**Format:** Use clear headings, bullet points, and numbered lists. Be comprehensive but concise. Use academic tone suitable for study materials."""),
                ("human", """Summarize these lecture notes for effective studying:

{text}

Create an organized summary with key concepts, definitions, and study tips.""")
            ])

            chain = prompt | self.llm
            response = chain.invoke({"text": text})
            return response.content
        except Exception as e:
            error_msg = str(e)
            if "maximum context length" in error_msg.lower() or "input too long" in error_msg.lower():
                # Truncate text further and retry
                truncated_text = text[:2000] + "..."
                try:
                    chain = prompt | self.llm
                    response = chain.invoke({"text": truncated_text})
                    return response.content + "\n\n(Note: Text was truncated due to length limitations)"
                except Exception:
                    return self._generate_fallback_summary(text)
            else:
                print(f"Error with OpenAI API for summarization: {e}")
                return self._generate_fallback_summary(text)

    def _generate_fallback_summary(self, text: str) -> str:
        """Generate a simple summary when OpenAI is not available"""
        if not text or len(text.strip()) == 0:
            return "No text content found to summarize."

        # Simple extractive summarization
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        if len(sentences) <= 3:
            return text  # Return original if already short

        # Score sentences based on length and position
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            # Prefer sentences in the middle and longer sentences
            position_score = 1.0 if 0.2 < i/len(sentences) < 0.8 else 0.5
            length_score = min(len(sentence.split()) / 20, 1.0)  # Prefer sentences with 10-20 words
            score = position_score * length_score
            scored_sentences.append((sentence, score))

        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = scored_sentences[:min(5, len(scored_sentences))]

        # Sort back by original position for coherence
        top_sentences.sort(key=lambda x: sentences.index(x[0]))

        summary = '. '.join([s[0] for s in top_sentences])
        return f"Summary: {summary}."

class TaskPlannerAgent:
    def __init__(self):
        try:
            # Use GPT-3.5-turbo for faster responses, fallback to GPT-4 if needed
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                api_key=api_key,
                temperature=0.2,  # Lower temperature for more consistent planning
                max_tokens=1200  # Reduced for faster responses
            )
        except Exception as e:
            try:
                print(f"GPT-3.5-turbo not available, trying GPT-4: {e}")
                self.llm = ChatOpenAI(
                    model="gpt-4",
                    api_key=api_key,
                    temperature=0.2,
                    max_tokens=1200
                )
            except Exception as e2:
                print(f"Warning: Failed to initialize OpenAI client: {e2}")
                self.llm = None

    def create_task_plan(self, assignment_description: str, due_date: str, steps: int = 5) -> List[Dict[str, str]]:
        """Break down assignment into subtasks with deadlines"""
        if self.llm is None:
            # Fallback implementation when OpenAI is not available
            return self._create_fallback_task_plan(assignment_description, due_date, steps)

        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert academic task planner who creates detailed, actionable, and intelligent task breakdowns for student assignments.

**Your Expertise:**
- Analyze assignment type, complexity, and requirements
- Create LOGICAL, CHRONOLOGICAL task sequences
- Break down complex assignments into manageable, specific steps
- Consider time management and realistic workloads
- Ensure tasks build progressively toward completion

**Task Planning Rules:**
1. **Analyze Deeply**: Understand the assignment's core requirements, scope, and challenges
2. **Logical Sequence**: Tasks must follow a natural progression (research → planning → execution → review)
3. **Specific & Actionable**: Each task should be concrete with clear deliverables
4. **Time-Appropriate**: Consider the complexity and time needed for each phase
5. **Progressive Difficulty**: Start with foundational tasks, build to complex ones
6. **Quality Focus**: Include review, revision, and quality assurance steps

**Output Format:** Return ONLY a valid JSON array like this:
[{{"title": "Specific Task Title", "description": "Detailed description of exactly what to do in this step", "deadline_percentage": 20}},
 {{"title": "Next Task Title", "description": "Detailed description of the next step", "deadline_percentage": 40}},
 ...]

**Percentage Guidelines:**
- Research/Planning: 20-30%
- Initial Work: 30-50%
- Main Execution: 50-70%
- Review/Revision: 70-90%
- Finalization: 90-100%"""),
                ("human", """Create an intelligent {steps}-step task breakdown for this assignment:

ASSIGNMENT: {assignment}
DUE DATE: {due_date}
REQUIRED STEPS: {steps}

Analyze this assignment carefully and create a logical, chronological sequence of specific, actionable tasks that will lead to successful completion. Each task should be detailed enough to know exactly what to do.

Return ONLY the JSON array with intelligent, assignment-specific tasks.""")
            ])

            chain = prompt | self.llm
            response = chain.invoke({
                "assignment": assignment_description,
                "due_date": due_date,
                "steps": steps
            })

            # Parse the JSON response
            import json
            try:
                # Clean the response content to extract JSON
                content = response.content.strip()
                # Remove any markdown formatting if present
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()

                tasks = json.loads(content)

                # Validate and ensure chronological order and proper percentages
                if isinstance(tasks, list) and len(tasks) > 0:
                    # Ensure chronological order by sorting by deadline_percentage
                    validated_tasks = []
                    for i, task in enumerate(tasks):
                        if isinstance(task, dict) and 'title' in task and 'description' in task:
                            # Convert deadline_percentage to int for sorting
                            try:
                                deadline_pct = int(task.get('deadline_percentage', (i+1) * (100 // len(tasks))))
                            except (ValueError, TypeError):
                                deadline_pct = (i+1) * (100 // len(tasks))

                            validated_task = {
                                'title': str(task['title']).strip(),
                                'description': str(task['description']).strip(),
                                'deadline_percentage': str(deadline_pct)
                            }
                            validated_tasks.append(validated_task)

                    # Sort by deadline percentage (ensure all are integers)
                    for task in validated_tasks:
                        try:
                            task['deadline_percentage'] = int(task['deadline_percentage'])
                        except (ValueError, TypeError):
                            task['deadline_percentage'] = 100  # Default to 100 if conversion fails

                    validated_tasks.sort(key=lambda x: x['deadline_percentage'])

                    if validated_tasks:
                        return validated_tasks

                # If validation fails, use fallback
                return self._create_fallback_task_plan(assignment_description, due_date, steps)

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"JSON parsing error in task planning: {e}, content: {response.content[:200]}...")
                # Fallback if JSON parsing fails
                return self._create_fallback_task_plan(assignment_description, due_date, steps)
        except Exception as e:
            print(f"Error with OpenAI API for task planning: {e}")
            return self._create_fallback_task_plan(assignment_description, due_date, steps)

    def _create_fallback_task_plan(self, assignment_description: str, due_date: str, steps: int = 5) -> List[Dict[str, str]]:
        """Create an intelligent task plan when OpenAI is not available"""
        tasks = []

        # Analyze assignment type and create appropriate chronological tasks
        assignment_lower = assignment_description.lower()

        if any(word in assignment_lower for word in ['essay', 'paper', 'report', 'write', 'writing']):
            # Writing assignment - chronological order
            task_templates = [
                ("Research and Topic Selection", "Research the topic thoroughly, gather reliable sources, and select a clear, focused thesis statement"),
                ("Create Detailed Outline", "Develop a comprehensive outline with main arguments, supporting evidence, and logical structure"),
                ("Write First Draft", "Write the complete first draft focusing on getting ideas down without worrying about perfection"),
                ("Revise Content and Structure", "Review and revise the content, improve organization, strengthen arguments, and ensure logical flow"),
                ("Edit and Proofread", "Edit for clarity, grammar, and style; proofread carefully for errors and formatting"),
                ("Final Review and Submission", "Do a final review, make any last adjustments, and submit the completed assignment")
            ]
        elif any(word in assignment_lower for word in ['presentation', 'slides', 'powerpoint', 'speech']):
            # Presentation assignment - chronological order
            task_templates = [
                ("Research Topic Thoroughly", "Gather comprehensive information, data, and examples about your presentation topic"),
                ("Organize Content Logically", "Structure your content into clear sections with main points and supporting details"),
                ("Create Visual Slides", "Design effective slides with clear visuals, bullet points, and minimal text"),
                ("Practice Delivery", "Rehearse your presentation multiple times, focusing on timing and smooth delivery"),
                ("Prepare for Questions", "Anticipate potential questions and prepare thoughtful answers"),
                ("Final Preparation", "Review all materials, test equipment, and ensure everything is ready for presentation day")
            ]
        elif any(word in assignment_lower for word in ['project', 'code', 'programming', 'software', 'development']):
            # Project/Programming assignment - chronological order
            task_templates = [
                ("Analyze Requirements", "Carefully read and understand all project requirements, constraints, and deliverables"),
                ("Design Solution Architecture", "Plan the overall approach, create diagrams, and design the system architecture"),
                ("Implement Core Features", "Write the main code and implement the primary functionality of your project"),
                ("Test and Debug", "Thoroughly test all features, identify and fix bugs, and ensure everything works correctly"),
                ("Add Documentation", "Create comprehensive documentation, comments, and user instructions"),
                ("Final Review and Submission", "Review the entire project, make final improvements, and prepare for submission")
            ]
        elif any(word in assignment_lower for word in ['exam', 'test', 'study', 'review', 'preparation']):
            # Study/Test preparation - chronological order
            task_templates = [
                ("Review All Course Materials", "Go through all notes, textbooks, and course materials to refresh your knowledge"),
                ("Identify Key Concepts", "Focus on the most important topics, formulas, and concepts that will be on the exam"),
                ("Practice with Sample Problems", "Work through practice problems and past exam questions to build confidence"),
                ("Review Weak Areas", "Identify and spend extra time on topics where you need more understanding"),
                ("Take Practice Tests", "Simulate exam conditions with timed practice tests to improve time management"),
                ("Final Quick Review", "Do a final review of key points, formulas, and strategies the day before the exam")
            ]
        elif any(word in assignment_lower for word in ['research', 'thesis', 'dissertation']):
            # Research assignment - chronological order
            task_templates = [
                ("Define Research Question", "Clearly define your research question and objectives for the study"),
                ("Conduct Literature Review", "Review existing research, identify gaps, and build theoretical framework"),
                ("Design Methodology", "Plan your research approach, methods, data collection, and analysis techniques"),
                ("Collect and Analyze Data", "Gather data according to your plan and perform thorough analysis"),
                ("Write Research Report", "Write clear, well-structured sections with proper citations and evidence"),
                ("Review and Finalize", "Get feedback, make revisions, ensure proper formatting, and finalize submission")
            ]
        else:
            # Generic assignment - chronological order
            task_templates = [
                ("Understand Requirements", "Carefully read and understand all assignment requirements and expectations"),
                ("Gather Information", "Research and collect all necessary information, resources, and materials"),
                ("Create Initial Plan", "Develop a clear plan and outline for completing the assignment"),
                ("Execute Main Work", "Complete the primary work according to your plan and requirements"),
                ("Review and Improve", "Review your work, identify areas for improvement, and make necessary changes"),
                ("Finalize and Submit", "Do final checks, ensure everything meets requirements, and submit on time")
            ]

        # Adjust to requested number of steps
        if len(task_templates) > steps:
            # Select evenly spaced tasks to maintain chronological order
            step_size = len(task_templates) / steps
            selected_templates = []
            for i in range(steps):
                idx = int(i * step_size)
                if idx < len(task_templates):
                    selected_templates.append(task_templates[idx])
            task_templates = selected_templates
        elif len(task_templates) < steps:
            # Add intermediate tasks while maintaining order
            expanded_templates = []
            for i, template in enumerate(task_templates):
                expanded_templates.append(template)
                if i < len(task_templates) - 1 and len(expanded_templates) < steps:
                    # Add intermediate task
                    mid_title = f"Continue {template[0].split()[0]} Phase"
                    mid_desc = f"Continue working on {template[0].lower()}"
                    expanded_templates.append((mid_title, mid_desc))
            task_templates = expanded_templates[:steps]

        # Create tasks with appropriate percentages (ensuring chronological order)
        total_tasks = len(task_templates)
        for i, (title, description) in enumerate(task_templates):
            # Calculate percentage to ensure they sum to 100% and maintain order
            if i == total_tasks - 1:
                # Last task gets remaining percentage to ensure sum = 100
                percentage = 100 - sum(int(task["deadline_percentage"]) for task in tasks)
            else:
                percentage = ((i + 1) * 100) // total_tasks

            tasks.append({
                "title": title,
                "description": description,
                "deadline_percentage": str(percentage)
            })

        return tasks

class QAAgent:
    def __init__(self):
        try:
            # Use GPT-3.5-turbo for faster responses, fallback to GPT-4 if needed
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                api_key=api_key,
                temperature=0.1,  # Lower temperature for more accurate answers
                max_tokens=1000  # Reduced for faster responses
            )
        except Exception as e:
            try:
                print(f"GPT-3.5-turbo not available, trying GPT-4: {e}")
                self.llm = ChatOpenAI(
                    model="gpt-4",
                    api_key=api_key,
                    temperature=0.1,
                    max_tokens=1000
                )
            except Exception as e2:
                print(f"Warning: Failed to initialize OpenAI client for Q&A: {e2}")
                self.llm = None

    def answer_question(self, question: str, context_documents: List[Dict[str, Any]]) -> str:
        """Answer questions using RAG with provided context"""
        if self.llm is None:
            # Fallback: simple text-based answer generation
            return self._generate_fallback_answer(question, context_documents)

        if not context_documents:
            return "I don't have any documents in my knowledge base yet. Please upload some study notes first so I can help answer your questions."

        try:
            # Prepare context from retrieved documents - limit to top 5 most relevant for better coverage
            sorted_docs = sorted(context_documents,
                               key=lambda x: x.get('similarity_score', x.get('relevance_score', 1 - x.get('distance', 0))),
                               reverse=True)

            # Take top 5 documents for comprehensive context
            top_docs = sorted_docs[:5]
            context_parts = []
            total_context_length = 0
            max_context_length = 6000  # Limit total context to prevent token overflow

            for i, doc in enumerate(top_docs, 1):
                content = doc.get('content', '').strip()
                if content:
                    # Truncate individual document content if too long
                    if len(content) > 1200:  # Limit each document to ~1200 chars
                        content = content[:1200] + "..."

                    # Check if adding this document would exceed total context limit
                    if total_context_length + len(content) > max_context_length:
                        remaining_space = max_context_length - total_context_length
                        if remaining_space > 200:  # Only add if we have meaningful space
                            content = content[:remaining_space] + "..."
                        else:
                            break  # Stop adding more documents

                    # Add source identifier with metadata
                    metadata = doc.get('metadata', {})
                    filename = metadata.get('filename', f'Document {i}')
                    source_info = f"[Source {i}: {filename}]"
                    context_parts.append(f"{source_info}\n{content}")
                    total_context_length += len(content)

            context_text = "\n\n".join(context_parts)

            if not context_text.strip():
                return "The uploaded documents don't contain readable text content. Please try uploading different PDF files."

            # Log context preparation for debugging
            print(f"🤖 Q&A Context prepared: {len(top_docs)} documents, {total_context_length} characters")

            # Detect if this is a general/overview question
            is_general_question = any(phrase in question.lower() for phrase in [
                "what is", "what's", "tell me about", "document about", "about this",
                "overview", "summary", "main topic", "key points", "content", "subject"
            ])

            if is_general_question:
                # For general questions, provide a comprehensive overview
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a study assistant that provides overviews of uploaded documents and study materials.

**IMPORTANT:** Base your entire response ONLY on the content provided in the "Study Materials Context" section below. Do not use any external knowledge or assumptions.

**Your Task:**
- Analyze the provided study materials context
- Provide a comprehensive overview of the main topics covered
- Summarize key concepts, important information, and main ideas
- Structure your response to be educational and helpful for studying

**Response Guidelines:**
- Start with the primary subject/topic of the document
- List and explain the main concepts and key points
- Be specific about what information is actually in the document
- Keep your response focused and well-organized
- If the document has multiple sections or topics, mention them clearly

**Remember:** Only discuss information that is explicitly present in the provided context."""),
                    ("human", """Please provide a comprehensive overview of the uploaded study materials by answering: {question}

**Study Materials Context (Use ONLY this information):**
{context}

Based solely on the content above, provide a clear and educational overview of what this document/material covers.""")
                ])
            else:
                # For specific questions, use the standard Q&A approach
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a study assistant that answers questions using ONLY the provided study materials context.

**CRITICAL:** Your answers must be based EXCLUSIVELY on the content in the "Study Materials Context" section. Do not use external knowledge, assumptions, or information not present in the provided context.

**Answering Rules:**
1. Use ONLY information from the provided context - ignore any external knowledge
2. If a question cannot be answered using the provided context, say: "Based on the provided study materials, I don't have enough information to answer this question."
3. When possible, quote or reference specific parts of the context
4. Be accurate and truthful to the content provided
5. Reference source documents when multiple sources are relevant

**Response Style:**
- Be direct and factual
- Include specific details from the materials
- Educational and helpful for studying
- Cite sources when information comes from multiple documents
- Stay focused on the actual content provided"""),
                    ("human", """**Study Materials Context (Answer using ONLY this information):**
{context}

**Question:** {question}

Answer the question using ONLY the study materials context provided above. If the information needed to answer this question is not present in the context, clearly state that limitation.""")
                ])

            chain = prompt | self.llm
            response = chain.invoke({
                "context": context_text,
                "question": question
            })

            answer = response.content.strip()

            # Validate that the answer is actually based on context
            if not answer or len(answer) < 10:
                return self._generate_fallback_answer(question, context_documents)

            # Check if answer indicates no information (good)
            if "don't have enough information" in answer.lower() or "not in the context" in answer.lower():
                return answer

            # If answer seems generic or not context-specific, use fallback
            generic_phrases = ["i don't know", "i'm not sure", "based on general knowledge", "from my training"]
            if any(phrase in answer.lower() for phrase in generic_phrases):
                return self._generate_fallback_answer(question, context_documents)

            return answer

        except Exception as e:
            print(f"Error with OpenAI API for Q&A: {e}")
            return self._generate_fallback_answer(question, context_documents)

    def _generate_fallback_answer(self, question: str, context_documents: List[Dict[str, Any]]) -> str:
        """Generate a simple answer when OpenAI is not available"""
        if not context_documents:
            return "I don't have any relevant information in my knowledge base to answer this question. Please upload some study notes first."

        # Sort documents by relevance score
        sorted_docs = sorted(context_documents,
                           key=lambda x: x.get('similarity_score', x.get('relevance_score', 1 - x.get('distance', 0))),
                           reverse=True)

        # Get the most relevant document
        best_doc = sorted_docs[0]
        content = best_doc.get('content', '')

        # Simple keyword extraction and answer generation
        question_lower = question.lower()
        content_lower = content.lower()

        # Extract key information based on question type
        if any(word in question_lower for word in ['what', 'define', 'explain', 'describe']):
            # Definition/explanation type question
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            # Look for sentences that might contain definitions or explanations
            relevant_sentences = []
            for sentence in sentences:
                if len(sentence.split()) > 10 and any(word in sentence.lower() for word in ['is', 'are', 'means', 'refers']):
                    relevant_sentences.append(sentence)

            if relevant_sentences:
                return f"Based on the study materials: {relevant_sentences[0]}"

        elif any(word in question_lower for word in ['how', 'process', 'steps', 'method']):
            # Process/steps type question
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            # Look for numbered lists or process descriptions
            process_sentences = []
            for sentence in sentences:
                if any(char.isdigit() for char in sentence[:10]) or any(word in sentence.lower() for word in ['first', 'then', 'next', 'finally']):
                    process_sentences.append(sentence)

            if process_sentences:
                return f"According to the documents: {'. '.join(process_sentences[:2])}"

        # Default: extract relevant sentences containing question keywords
        question_words = [word for word in question_lower.split() if len(word) > 3]
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        relevant_sentences = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in question_words):
                relevant_sentences.append(sentence)

        if relevant_sentences:
            answer = '. '.join(relevant_sentences[:2])
            confidence = best_doc.get('similarity_score', best_doc.get('relevance_score', 0))
            confidence_text = "high" if confidence > 0.7 else "moderate" if confidence > 0.5 else "low"
            return f"Based on the study materials (confidence: {confidence_text}): {answer}."
        else:
            # Return a portion of the most relevant document
            return f"From the most relevant study material: {content[:400]}..."

# Create agent instances
note_summarizer = NoteSummarizerAgent()
task_planner = TaskPlannerAgent()
qa_agent = QAAgent()
