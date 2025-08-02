from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import numpy as np
import faiss
import re
import torch
import json
from typing import Dict, List, Tuple, Optional

class EnhancedTextIndex:
    def __init__(self):
        self.embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        

        self.qa_model = pipeline(
            'question-answering', 
            model='microsoft/DialoGPT-medium',  
            tokenizer='microsoft/DialoGPT-medium',
            return_all_scores=True
        )
        
        # Uncomment below for even better performance (requires more resources)
        # self.qa_model = pipeline(
        #     'question-answering',
        #     model='deepset/deberta-v3-large-squad2',
        #     tokenizer='deepset/deberta-v3-large-squad2'
        # )
        
        self.index = None
        self.text_chunks = []
        self.page_mapping = []
        self.chunk_metadata = []  

        self.resume_patterns = {
            'name': [
                r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',  
                r'Name:\s*([A-Za-z\s]+)',
                r'([A-Z][a-z]+\s+[A-Z][a-z]+)',  
            ],
            'email': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            ],
            'phone': [
                r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                r'(\+?\d{1,3}[-.\s]?)?\d{10}',
            ],
            'education': [
                r'(Bachelor|Master|PhD|B\.?S\.?|M\.?S\.?|B\.?A\.?|M\.?A\.?|B\.?Tech|M\.?Tech)',
                r'(University|College|Institute|School)',
                r'(Degree|Graduation|Education)',
            ],
            'experience': [
                r'(Experience|Work|Employment|Career)',
                r'(\d{4}[-\s](?:present|current|\d{4}))',  
                r'(years?|months?)\s+(?:of\s+)?(?:experience|work)',
            ],
            'skills': [
                r'(Skills|Technologies|Programming|Languages)',
                r'(Python|Java|JavaScript|React|Node|SQL|AWS|Docker)',
            ]
        }

    def extract_resume_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract structured information from resume text"""
        entities = {
            'names': [],
            'emails': [],
            'phones': [],
            'education': [],
            'experience': [],
            'skills': []
        }
        
        for pattern in self.resume_patterns['email']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['emails'].extend(matches)
        
        for pattern in self.resume_patterns['phone']:
            matches = re.findall(pattern, text)
            entities['phones'].extend(matches)
        
        lines = text.split('\n')[:5] 
        for line in lines:
            for pattern in self.resume_patterns['name']:
                matches = re.findall(pattern, line.strip())
                if matches:
                    entities['names'].extend(matches)
        
        education_text = ""
        for line in text.split('\n'):
            if any(keyword.lower() in line.lower() for keyword in ['education', 'degree', 'university', 'college']):
                education_text += line + " "
        
        for pattern in self.resume_patterns['education']:
            matches = re.findall(pattern, education_text, re.IGNORECASE)
            entities['education'].extend(matches)
        
        experience_text = ""
        for line in text.split('\n'):
            if any(keyword.lower() in line.lower() for keyword in ['experience', 'work', 'employment', 'career']):
                experience_text += line + " "
        
        for pattern in self.resume_patterns['experience']:
            matches = re.findall(pattern, experience_text, re.IGNORECASE)
            entities['experience'].extend(matches)
        
        skills_text = ""
        for line in text.split('\n'):
            if any(keyword.lower() in line.lower() for keyword in ['skills', 'technologies', 'programming']):
                skills_text += line + " "
        
        for pattern in self.resume_patterns['skills']:
            matches = re.findall(pattern, skills_text, re.IGNORECASE)
            entities['skills'].extend(matches)
        
        for key in entities:
            entities[key] = list(set([item.strip() for item in entities[key] if item.strip()]))
        
        return entities

    def intelligent_chunk_resume(self, text: str, max_chunk_size: int = 500) -> List[Dict]:
        """Intelligent chunking specifically designed for resumes"""
        chunks = []
        
        section_headers = [
            r'(?i)(experience|work\s+experience|employment|career)',
            r'(?i)(education|academic|qualification)',
            r'(?i)(skills|technical\s+skills|technologies)',
            r'(?i)(projects|personal\s+projects)',
            r'(?i)(achievements|awards|accomplishments)',
            r'(?i)(certifications?|licenses?)',
            r'(?i)(summary|objective|profile)',
            r'(?i)(contact|personal\s+(?:details|information))'
        ]
        
        lines = text.split('\n')
        current_section = "header"
        current_chunk = ""
        section_start_idx = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            new_section = None
            for header_pattern in section_headers:
                if re.match(header_pattern, line):
                    new_section = re.match(header_pattern, line).group(1).lower()
                    break
            
            if new_section and current_chunk.strip():
                chunk_info = {
                    'text': current_chunk.strip(),
                    'section': current_section,
                    'start_line': section_start_idx,
                    'end_line': i-1
                }
                chunks.append(chunk_info)
                current_chunk = line
                current_section = new_section
                section_start_idx = i
            else:
                current_chunk += " " + line
                
                if len(current_chunk) > max_chunk_size:
                    chunk_info = {
                        'text': current_chunk.strip(),
                        'section': current_section,
                        'start_line': section_start_idx,
                        'end_line': i
                    }
                    chunks.append(chunk_info)
                    current_chunk = ""
                    section_start_idx = i + 1
        
        if current_chunk.strip():
            chunk_info = {
                'text': current_chunk.strip(),
                'section': current_section,
                'start_line': section_start_idx,
                'end_line': len(lines)-1
            }
            chunks.append(chunk_info)
        
        return chunks

    def create_index(self, text_dict: Dict[str, str]):
        """Create enhanced searchable index from PDF text with resume-specific processing"""
        self.text_chunks = []
        self.page_mapping = []
        self.chunk_metadata = []
        full_text = " ".join(text_dict.values())
        
        resume_entities = self.extract_resume_entities(full_text)
        
        for page_num, text in text_dict.items():
            if text.strip():
                page_chunks = self.intelligent_chunk_resume(text)
                
                for chunk_info in page_chunks:
                    self.text_chunks.append(chunk_info['text'])
                    self.page_mapping.append(page_num)
                    
                    metadata = {
                        'page': page_num,
                        'section': chunk_info['section'],
                        'entities': resume_entities,
                        'chunk_type': 'resume_section'
                    }
                    self.chunk_metadata.append(metadata)
        
        if not self.text_chunks:
            raise ValueError("No valid text chunks created from PDF")
        
        embeddings = self.embedder.encode(self.text_chunks).astype('float32')
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  
        
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        print(f"Created enhanced index with {len(self.text_chunks)} chunks from {len(text_dict)} pages")
        print(f"Extracted entities: {json.dumps({k: v[:3] for k, v in resume_entities.items()}, indent=2)}")

    def get_direct_answer(self, question: str) -> Optional[str]:
        """Get direct answers for common resume questions using extracted entities"""
        question_lower = question.lower()
        
        if not self.chunk_metadata:
            return None
            
        entities = self.chunk_metadata[0]['entities']
        
        if any(word in question_lower for word in ['name', 'called', 'who']):
            if entities['names']:
                return f"The person's name is {entities['names'][0]}."
        
        elif any(word in question_lower for word in ['email', 'contact', 'reach']):
            if entities['emails']:
                return f"Email: {entities['emails'][0]}"
        
        elif any(word in question_lower for word in ['phone', 'number', 'call']):
            if entities['phones']:
                return f"Phone: {entities['phones'][0]}"
        
        elif any(word in question_lower for word in ['education', 'study', 'degree', 'university', 'college']):
            if entities['education']:
                return f"Education includes: {', '.join(entities['education'][:3])}"
        
        elif any(word in question_lower for word in ['skills', 'technology', 'programming', 'language']):
            if entities['skills']:
                return f"Skills include: {', '.join(entities['skills'][:5])}"
        
        return None

    def query(self, question: str) -> str:
        """Enhanced query with better context retrieval and reduced hallucination"""
        if self.index is None or not self.text_chunks:
            return "No document indexed yet."
        
        try:
            direct_answer = self.get_direct_answer(question)
            if direct_answer:
                return direct_answer
            
            q_emb = self.embedder.encode([question]).astype('float32')
            faiss.normalize_L2(q_emb)
            
            D, I = self.index.search(q_emb, k=5)  
            
            relevant_chunks = []
            relevant_metadata = []
            
            for i, idx in enumerate(I[0]):
                similarity_score = D[0][i]
                if similarity_score > 0.3: 
                    chunk_text = self.text_chunks[idx]
                    metadata = self.chunk_metadata[idx]
                    
                    relevant_chunks.append({
                        'text': chunk_text,
                        'page': metadata['page'],
                        'section': metadata['section'],
                        'score': similarity_score
                    })
                    relevant_metadata.append(metadata)
            
            if not relevant_chunks:
                return "I couldn't find relevant information in the document to answer your question. Please try rephrasing your question or ask about specific resume sections like experience, education, or skills."
            
            relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
            
            context_parts = []
            for chunk in relevant_chunks[:3]:  
                section_info = f"[{chunk['section'].title()} Section, Page {chunk['page']}]"
                context_parts.append(f"{section_info}\n{chunk['text']}")
            
            context = "\n\n".join(context_parts)
            
            try:
                result = self.qa_model(
                    question=f"Based on this resume content, {question}",
                    context=context,
                    max_answer_len=200,
                    handle_impossible_answer=True
                )
                
                if isinstance(result, list):
                    result = result[0] if result else {}
                
                answer = result.get('answer', '').strip()
                confidence = result.get('score', 0)
                
                if confidence < 0.2 or not answer or len(answer) < 3:
                    section_summaries = []
                    for chunk in relevant_chunks[:2]:
                        section_summaries.append(f"**{chunk['section'].title()}**: {chunk['text'][:150]}...")
                    
                    return f"Based on the resume, here's the relevant information I found:\n\n" + "\n\n".join(section_summaries)
                
                primary_section = relevant_chunks[0]['section']
                return f"{answer}\n\n*Source: {primary_section.title()} section*"
                
            except Exception as qa_error:
                print(f"QA model error: {qa_error}")
                return self._create_fallback_response(relevant_chunks, question)
            
        except Exception as e:
            print(f"Error in query: {str(e)}")
            return "Sorry, I encountered an error while processing your question. Please try again with a more specific question."

    def _create_fallback_response(self, relevant_chunks: List[Dict], question: str) -> str:
        """Create a structured response when QA model fails"""
        question_lower = question.lower()
        
        sections = {}
        for chunk in relevant_chunks:
            section = chunk['section']
            if section not in sections:
                sections[section] = []
            sections[section].append(chunk['text'])
        
        response_parts = []
        for section, texts in sections.items():
            combined_text = " ".join(texts)[:300]  
            response_parts.append(f"**{section.title()}**: {combined_text}")
        
        return f"Based on your question about the resume, here's the relevant information:\n\n" + "\n\n".join(response_parts)

    def get_resume_summary(self) -> Dict:
        """Get a structured summary of the resume"""
        if not self.chunk_metadata:
            return {"error": "No resume indexed"}
        
        entities = self.chunk_metadata[0]['entities']
        sections = set(meta['section'] for meta in self.chunk_metadata)
        
        return {
            "name": entities['names'][0] if entities['names'] else "Not found",
            "email": entities['emails'][0] if entities['emails'] else "Not found",
            "phone": entities['phones'][0] if entities['phones'] else "Not found",
            "sections_found": list(sections),
            "total_chunks": len(self.text_chunks),
            "pages": len(set(self.page_mapping))
        }

    def answer_specific_question(self, question: str) -> str:
        """Handle specific resume questions with high accuracy"""
        question_lower = question.lower()
        
        if not self.chunk_metadata:
            return "No resume data available."
        
        entities = self.chunk_metadata[0]['entities']
        
        if 'name' in question_lower:
            names = entities.get('names', [])
            if names:
                return f"The person's name is {names[0]}."
            else:
                return "I couldn't extract the name from the resume. It might be in an image or unusual format."
        
        elif 'education' in question_lower or 'degree' in question_lower:
            education = entities.get('education', [])
            if education:
                return f"Education background includes: {', '.join(education)}"
            else:
                education_chunks = [
                    chunk for i, chunk in enumerate(self.text_chunks) 
                    if self.chunk_metadata[i]['section'] == 'education'
                ]
                if education_chunks:
                    return f"Education information: {education_chunks[0][:200]}..."
                return "No clear education information found in the resume."
        
        elif 'experience' in question_lower or 'work' in question_lower:
            experience_chunks = [
                chunk for i, chunk in enumerate(self.text_chunks)
                if self.chunk_metadata[i]['section'] in ['experience', 'work']
            ]
            if experience_chunks:
                return f"Work experience: {experience_chunks[0][:300]}..."
            return "No work experience section clearly identified."
        
        elif 'skills' in question_lower:
            skills = entities.get('skills', [])
            if skills:
                return f"Technical skills include: {', '.join(skills)}"
            else:
                skills_chunks = [
                    chunk for i, chunk in enumerate(self.text_chunks)
                    if self.chunk_metadata[i]['section'] == 'skills'
                ]
                if skills_chunks:
                    return f"Skills: {skills_chunks[0][:200]}..."
                return "No skills section clearly identified."
        
        elif 'contact' in question_lower:
            contact_info = []
            if entities.get('emails'):
                contact_info.append(f"Email: {entities['emails'][0]}")
            if entities.get('phones'):
                contact_info.append(f"Phone: {entities['phones'][0]}")
            
            if contact_info:
                return "Contact information:\n" + "\n".join(contact_info)
            return "No contact information clearly extracted."
        
        return self.query(question)

    def get_stats(self) -> Dict:
        """Get comprehensive index statistics"""
        if not self.chunk_metadata:
            return {"indexed": False}
        
        entities = self.chunk_metadata[0]['entities']
        sections = [meta['section'] for meta in self.chunk_metadata]
        
        return {
            "total_chunks": len(self.text_chunks),
            "indexed": self.index is not None,
            "pages": len(set(self.page_mapping)),
            "sections": list(set(sections)),
            "extracted_entities": {
                "names_found": len(entities.get('names', [])),
                "emails_found": len(entities.get('emails', [])),
                "phones_found": len(entities.get('phones', [])),
                "education_items": len(entities.get('education', [])),
                "skills_found": len(entities.get('skills', []))
            }
        }