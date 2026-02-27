"""
generator.py
------------
Memanggil LLM dan mengembalikan analisis terstruktur.

Mendukung empat provider:
  - Groq         (GRATIS, cepat — gunakan untuk testing/development)
  - HuggingFace  (GRATIS, via Inference API)
  - Anthropic Claude (berbayar)
  - OpenAI GPT-4o    (berbayar)

Atur via .env:
    LLM_PROVIDER=groq        → gunakan Groq (llama-3.3-70b, GRATIS)
    LLM_PROVIDER=huggingface → gunakan HuggingFace Inference API
    LLM_PROVIDER=anthropic   → gunakan Claude
    LLM_PROVIDER=openai      → gunakan OpenAI GPT-4o
"""

import os
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER        = os.getenv("LLM_PROVIDER", "groq").lower()
CLAUDE_MODEL        = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
OPENAI_MODEL        = os.getenv("OPENAI_MODEL", "gpt-4o")
GROQ_MODEL          = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
HUGGINGFACE_MODEL   = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """Struktur data untuk hasil analisis disinformasi/defamasi/ujaran kebencian."""
    classification: str = "UNKNOWN"
    confidence: str = "LOW"
    confidence_score: float = 0.0
    reasoning: str = ""
    evidence: list[str] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)
    recommendation: str = ""
    raw_response: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def display(self) -> str:
        """Format hasil analisis menjadi teks yang mudah dibaca."""
        sep = "=" * 60
        lines = [
            sep,
            "HASIL ANALISIS KONTEN",
            sep,
            f"Klasifikasi  : {self.classification}",
            f"Kepercayaan  : {self.confidence} ({self.confidence_score:.0%})",
            "",
            "Penjelasan:",
            f"   {self.reasoning}",
            "",
        ]

        if self.evidence:
            lines.append("Bukti Pendukung:")
            for i, e in enumerate(self.evidence, 1):
                lines.append(f"   [{i}] {e}")
            lines.append("")

        if self.sources_used:
            lines.append(f"Referensi Digunakan: {', '.join(self.sources_used)}")
            lines.append("")

        if self.recommendation:
            lines.append(f"Rekomendasi: {self.recommendation}")

        if self.error:
            lines.append(f"\nError: {self.error}")

        lines.append(sep)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Generator class
# ─────────────────────────────────────────────────────────────────────────────

class Generator:
    """
    Modul generator yang memanggil LLM (Claude atau GPT-4o) dan memparse hasil.
    Provider dipilih via LLM_PROVIDER env var: 'anthropic' atau 'openai'.
    """

    def __init__(self, provider: str = None):
        self.provider = (provider or LLM_PROVIDER).lower()
        self._client = None

        if self.provider == "groq":
            self._init_groq()
        elif self.provider == "anthropic":
            self._init_anthropic()
        elif self.provider == "openai":
            self._init_openai()
        elif self.provider == "huggingface":
            self._init_huggingface()
        else:
            raise ValueError(f"Provider tidak dikenal: '{self.provider}'. Pilih: groq, anthropic, openai, huggingface")

    def _init_groq(self):
        if not GROQ_API_KEY:
            raise EnvironmentError(
                "GROQ_API_KEY tidak ditemukan di .env\n"
                "Daftar gratis di: https://console.groq.com → API Keys"
            )
        from langchain_groq import ChatGroq
        self._llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)
        self.model = GROQ_MODEL
        print(f"[Generator] Provider: Groq (GRATIS) | Model: {self.model}")

    def _init_anthropic(self):
        if not ANTHROPIC_API_KEY:
            raise EnvironmentError("ANTHROPIC_API_KEY tidak ditemukan di .env")
        from langchain_anthropic import ChatAnthropic
        self._llm = ChatAnthropic(api_key=ANTHROPIC_API_KEY, model_name=CLAUDE_MODEL)
        self.model = CLAUDE_MODEL
        print(f"[Generator] Provider: Anthropic (LangChain) | Model: {self.model}")

    def _init_openai(self):
        if not OPENAI_API_KEY:
            raise EnvironmentError("OPENAI_API_KEY tidak ditemukan di .env")
        from langchain_openai import ChatOpenAI
        self._llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)
        self.model = OPENAI_MODEL
        print(f"[Generator] Provider: OpenAI (LangChain) | Model: {self.model}")

    def _init_huggingface(self):
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        import os
        if HUGGINGFACE_API_KEY:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_KEY
            
        endpoint = HuggingFaceEndpoint(
            repo_id=HUGGINGFACE_MODEL, 
            task="text-generation",
            huggingfacehub_api_token=HUGGINGFACE_API_KEY or None,
            return_full_text=False
        )
        self._llm = ChatHuggingFace(llm=endpoint)
        self.model = HUGGINGFACE_MODEL
        print(f"[Generator] Provider: HuggingFace (LangChain) | Model: {self.model}")

    # ------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1500,
        temperature: float = 0.1,
    ) -> AnalysisResult:
        """
        Kirim prompt ke LLM didukung LangChain dan kembalikan AnalysisResult terstruktur.
        """
        print(f"[Generator] Mengirim request ke {self.model} via LangChain...")

        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            if self.provider == "huggingface":
                self._llm.llm.model_kwargs = {"max_new_tokens": max_tokens}
                self._llm.llm.temperature = max(0.01, temperature)
                response = self._llm.invoke(messages)
            else:
                response = self._llm.invoke(messages, temperature=temperature, max_tokens=max_tokens)

            raw_text = response.content
            print(f"[Generator] Response diterima ({len(raw_text)} karakter).")
            return self._parse_response(raw_text)

        except Exception as e:
            print(f"[Generator] Error: {e}")
            return AnalysisResult(classification="ERROR", error=str(e), raw_response="")

    def _parse_response(self, raw_text: str) -> AnalysisResult:
        """Parse JSON dari respons LLM, handle markdown code blocks."""
        # Coba ekstrak JSON dari markdown block
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fallback: cari kurung kurawal pertama dan matching kurung penutupnya
            start_idx = raw_text.find('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = -1
                for i in range(start_idx, len(raw_text)):
                    if raw_text[i] == '{':
                        brace_count += 1
                    elif raw_text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i
                            break
                
                if end_idx != -1:
                    json_str = raw_text[start_idx:end_idx+1]
                else:
                    json_str = raw_text.strip()
            else:
                json_str = raw_text.strip()

        try:
            data = json.loads(json_str)
            return AnalysisResult(
                classification=data.get("classification", "UNKNOWN"),
                confidence=data.get("confidence", "LOW"),
                confidence_score=float(data.get("confidence_score", 0.0)),
                reasoning=data.get("reasoning", ""),
                evidence=data.get("evidence", []),
                sources_used=data.get("sources_used", []),
                recommendation=data.get("recommendation", ""),
                raw_response=raw_text,
            )
        except json.JSONDecodeError as e:
            print(f"[Generator] Gagal parse JSON: {e}")
            return AnalysisResult(
                classification="PARSE_ERROR",
                reasoning=raw_text,
                raw_response=raw_text,
                error=f"JSON parse error: {e}",
            )
