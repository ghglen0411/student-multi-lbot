import os
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from openai import OpenAI
from google import genai
import anthropic
from perplexity import Perplexity


# =========================
# 1. 환경 변수 및 기본 설정
# =========================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

if not (OPENAI_API_KEY and GEMINI_API_KEY and ANTHROPIC_API_KEY and PERPLEXITY_API_KEY):
    raise RuntimeError(".env에 모든 API 키(OPENAI, GEMINI, ANTHROPIC, PERPLEXITY)가 설정되어 있는지 확인하세요.")


# =========================
# 2. 상수 및 공통 타입 정의
# =========================

ModeType = Literal["A", "B", "C", "D"]

GPT_MODEL_NAME = "gpt-4o-mini"
GEMINI_MODEL_NAME = "gemini-2.0-flash"
CLAUDE_MODEL_NAME = "claude-3-haiku-20240307"
PERPLEXITY_MODEL_NAME = "sonar"

SYSTEM_PROMPT_GPT = "너는 고등학생 과제를 도와주는 한국어 튜터야."

REPORT_KEYWORDS = (
    "보고서",
    "보고 서",
    "보고서를",
    "보고서로",
    "작성해",
    "작성해줘",
    "작성해 줘",
)


# =========================
# 3. 각 서비스 클라이언트 초기화
# =========================

openai_client = OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
perplexity_client = Perplexity(api_key=PERPLEXITY_API_KEY)

# 🔁 A 모드의 "1단계 탐구 방향 결과"를 저장해 둘 전역 변수
last_plan_for_a: Optional[str] = None


# =========================
# 4. FastAPI 기본 설정
# =========================

app = FastAPI(title="고등학생 맞춤형 멀티 챗봇")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 개발 단계에서는 * 허용, 배포 시에는 도메인 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_front_page_html() -> str:
    """
    루트 페이지에서 보여줄 간단한 웹 UI HTML.
    (입력 → /chat 호출 → 답변 표시)
    """
    return """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
      <meta charset="UTF-8" />
      <title>학생 과제 멀티 LLM 챗봇</title>
      <style>
        body {
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          max-width: 900px;
          margin: 0 auto;
          padding: 20px;
          background: #020617;
          color: #e5e7eb;
        }
        h1 {
          text-align: center;
          margin-bottom: 8px;
        }
        .subtitle {
          text-align: center;
          color: #9ca3af;
          font-size: 13px;
          margin-bottom: 18px;
        }
        .card {
          background: #0b1220;
          border-radius: 14px;
          padding: 16px;
          margin-top: 16px;
          border: 1px solid #1f2937;
        }
        label {
          display: block;
          margin-top: 8px;
          margin-bottom: 4px;
          font-weight: 600;
          font-size: 14px;
        }
        select, textarea, button {
          width: 100%;
          padding: 8px;
          border-radius: 8px;
          border: 1px solid #374151;
          background: #020617;
          color: #e5e7eb;
          box-sizing: border-box;
          font-size: 14px;
        }
        textarea {
          min-height: 130px;
          resize: vertical;
        }
        button {
          margin-top: 12px;
          cursor: pointer;
          font-weight: 600;
        }
        button:hover {
          background: #111827;
        }
        pre {
          white-space: pre-wrap;
          word-wrap: break-word;
          background: #020617;
          padding: 12px;
          border-radius: 8px;
          margin-top: 12px;
          max-height: 420px;
          overflow-y: auto;
          border: 1px solid #1f2937;
          font-size: 14px;
        }
        small {
          color: #9ca3af;
          font-size: 12px;
        }
      </style>
    </head>
    <body>
      <h1>학생 과제 멀티 LLM 챗봇</h1>
      <div class="subtitle">GPT + Gemini + Claude + Perplexity 오케스트레이션</div>

      <div class="card">
        <label for="mode">과제 유형 선택</label>
        <select id="mode">
          <option value="A">A. 탐구보고서 작성</option>
          <option value="B">B. 주장/논설문 작성</option>
          <option value="C" selected>C. 창의적 아이디어</option>
          <option value="D">D. 자료 조사 및 요약</option>
        </select>
        <small>모드에 따라 내부에서 사용하는 AI 조합이 달라져.</small>

        <label for="user_input">요청 내용 / 과제 설명</label>
        <textarea id="user_input" placeholder="예) 환경 관련 주제로 과학 탐구 아이디어를 만들어줘. 실험 가능해야 하고 보고서까지 쓸 수 있으면 좋겠어."></textarea>

        <button id="send_btn">챗봇에게 보내기</button>

        <pre id="answer_box">여기에 챗봇 답변이 표시됩니다.</pre>
      </div>

      <script>
        const API_URL = "http://127.0.0.1:8000/chat";

        async function sendRequest() {
          const mode = document.getElementById("mode").value;
          const userInput = document.getElementById("user_input").value.trim();
          const answerBox = document.getElementById("answer_box");
          const btn = document.getElementById("send_btn");

          if (!userInput) {
            alert("요청 내용을 입력해 주세요!");
            return;
          }

          btn.disabled = true;
          btn.textContent = "생각 중...";
          answerBox.textContent = "여러 AI 모델(GPT, Gemini, Claude, Perplexity)을 오케스트레이션하는 중...";

          try {
            const res = await fetch(API_URL, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                mode: mode,
                user_input: userInput
              })
            });

            if (!res.ok) {
              const text = await res.text();
              throw new Error("서버 오류: " + text);
            }

            const data = await res.json();
            answerBox.textContent = data.answer;
          } catch (err) {
            console.error(err);
            answerBox.textContent = "에러가 발생했습니다: " + err.message;
          } finally {
            btn.disabled = false;
            btn.textContent = "챗봇에게 보내기";
          }
        }

        document.getElementById("send_btn").addEventListener("click", sendRequest);

        // ⌘+Enter 또는 Ctrl+Enter 로 보내기
        document.getElementById("user_input").addEventListener("keydown", (e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            sendRequest();
          }
        });
      </script>
    </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    """루트 페이지: 간단한 웹 UI 반환."""
    return build_front_page_html()


# =========================
# 5. 요청/응답 스키마
# =========================

class ChatRequest(BaseModel):
    mode: ModeType  # A:탐구, B:논설, C:아이디어, D:자료조사
    user_input: str


class ChatResponse(BaseModel):
    answer: str


# =========================
# 6. 공통 LLM 호출 함수들
# =========================

def call_gpt(prompt: str) -> str:
    """
    GPT 호출: 창의적 아이디어, 초안 작성 등.
    입력(prompt) → GPT 응답 텍스트
    """
    resp = openai_client.chat.completions.create(
        model=GPT_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_GPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
    )
    return resp.choices[0].message.content


def call_gemini(prompt: str) -> str:
    """
    Gemini 호출: 오케스트레이션 / 검증 / 정리.
    """
    resp = gemini_client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt,
    )
    return resp.text


def call_claude(prompt: str) -> str:
    """
    Claude 호출: 문장 다듬기·긴 글 작성용.
    """
    message = anthropic_client.messages.create(
        model=CLAUDE_MODEL_NAME,
        max_tokens=1500,
        temperature=0.7,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return "".join(block.text for block in message.content)


def call_perplexity(prompt: str) -> str:
    """
    Perplexity 호출: 자료 조사/웹 검색 (sonar 모델 사용).
    """
    completion = perplexity_client.chat.completions.create(
        model=PERPLEXITY_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return completion.choices[0].message.content


# =========================
# 7. 헬퍼 함수들
# =========================

def is_report_request(text: str) -> bool:
    """
    A 모드에서 사용자의 입력이 '보고서 작성' 의도인지 판별.
    단순 키워드 포함 여부로 판단.
    """
    return any(keyword in text for keyword in REPORT_KEYWORDS)


# =========================
# 8. 모드별 오케스트레이션 로직
# =========================

def handle_mode_a_plan(user_input: str) -> str:
    """
    A-1단계: 탐구 방향/계획 설계
    Perplexity: 배경 조사 → GPT: 탐구 방향 여러 개 → Gemini: 정리/추천
    (보고서 본문은 쓰지 않음)
    """
    # 1) 배경 지식
    bg = call_perplexity(
        f"""
고등학생 수준에서 이해할 수 있게, 아래 탐구 주제의 배경지식을 정리해줘.

탐구 주제: {user_input}
"""
    )

    # 2) GPT로 탐구 방향/가설 아이디어 여러 개
    ideas = call_gpt(
        f"""
[탐구 주제]
{user_input}

[배경지식]
{bg}

위 정보를 바탕으로, 고등학생 수행평가용 탐구 과제 후보를 설계해줘.
- 탐구 질문/가설
- 탐구 방법(실험 또는 조사)
- 난이도(쉬움/보통/어려움)

이런 아이디어를 3~5개 번호 목록(1, 2, 3, ...)으로 작성해줘.
"""
    )

    # 3) Gemini가 아이디어 중 1~2개를 추천 + 정리
    plan = call_gemini(
        f"""
너는 고등학생 탐구보고서 지도 교사야.

[탐구 주제]
{user_input}

[배경지식 요약]
{bg}

[가능한 탐구 방향 아이디어들]
{ideas}

위 내용을 바탕으로,
1. 가장 적절해 보이는 탐구 방향 1~2개를 추천하고,
2. 각 방향에 대해 다음 형식으로만 정리해줘.

- 탐구 방향 번호 (예: 후보 2번)
- 탐구 제목
- 탐구 목적
- 탐구 질문/가설
- 간단한 탐구 방법 개요

아직 보고서 본문을 쓰지는 말고,
'어떤 방향으로 탐구할지'만 정리해줘.

마지막 줄에 다음 문장을 꼭 붙여줘:

\"➡ 이 중에서 선택한 번호와 함께 '이제 이 중에서 X번으로 탐구 보고서 작성해줘'라고 요청하면, 이에 맞는 탐구 보고서를 작성해 줄 수 있습니다.\"
"""
    )

    return plan


def handle_mode_a_report(user_input: str, previous_answer: str) -> str:
    """
    A-2단계: 1단계에서 만든 탐구 방향/계획 + 사용자의 추가 요구를 기반으로
    실제 탐구 보고서 본문 작성.
    """
    # 1) Gemini: 사용자가 선택한 방향 파악 + 개요 생성
    outline = call_gemini(
        f"""
너는 고등학생 탐구보고서 작성 도우미야.

[이전 단계에서 생성된 탐구 방향/계획]
{previous_answer}

[사용자의 추가 요청]
{user_input}

1. 사용자가 어느 탐구 방향(몇 번)을 선택했는지 추론하고,
2. 그 방향에 맞는 탐구보고서의 '개요(목차 + 각 항목 요약)'를 먼저 만들어줘.

개요 형식 예시:
1. 서론 (문제 제기, 탐구 동기)
2. 이론적 배경
3. 탐구 방법
4. 탐구 결과
5. 결론 및 느낀 점

각 항목 아래에, 어떤 내용을 쓸지 2~3문장 정도로 간단히 설명해줘.
아직 문단 전체를 길게 쓰지는 마.
"""
    )

    # 2) Claude: 실제 보고서 본문 작성
    final_report = call_claude(
        f"""
아래는 고등학생 과학 탐구보고서의 개요야.

[보고서 개요]
{outline}

이 개요를 바탕으로, 실제 탐구보고서 초안을 작성해줘.
요구사항:
- 고등학생이 제출하는 수행평가/탐구보고서 톤
- 서론 / 이론적 배경 / 탐구 방법 / 탐구 결과(예상 결과 가능) / 결론 및 느낀 점 순서
- 각 항목은 최소 2~3문단 정도로 작성
- 너무 전문 용어만 남발하지 말고, 필요하면 괄호 안에 간단한 설명

전체를 하나의 보고서처럼 자연스럽게 이어지게 써줘.
"""
    )

    return final_report


def handle_mode_b_essay(user_input: str) -> str:
    """
    B. 논설문 모드:
    Perplexity: 사실/사례 수집 → GPT: 논설문 초안 → Gemini: 논리 체크 →
    Claude: 최종 문체/흐름 다듬기
    """
    # 1) Perplexity로 논거 자료 수집
    facts = call_perplexity(
        f"""
아래 논설문 주제와 관련해서, 사실 자료/통계/사례를 한국어로 간단히 정리해줘.
논설문 주제: {user_input}
"""
    )

    # 2) GPT로 논설문 초안 작성
    draft = call_gpt(
        f"""
너는 고등학생 논설문을 도와주는 한국어 선생님이야.

[논설문 주제]
{user_input}

[참고 자료]
{facts}

요구 사항:
- 서론 / 본론(2~3개의 논거) / 결론 구조
- 문어체, 존댓말
- 분량: 대략 800~1200자

위 조건에 맞는 논설문 초안을 작성해줘.
"""
    )

    # 3) Gemini로 논리 체크 & 약간 수정
    checked = call_gemini(
        f"""
다음은 고등학생이 제출할 논설문 초안이야.

[초안]
{draft}

너의 역할:
1. 논리 전개가 자연스러운지 확인하고, 어색하거나 모순된 부분을 지적 후 수정.
2. 과도한 표현이나 사실과 다를 수 있는 부분은 완화해서 표현.
3. 전체 구조(서론-본론-결론)는 유지하되, 문장의 흐름만 살짝 정리.

수정된 논설문만 한국어로 보여줘.
"""
    )

    # 4) Claude로 최종 문체/유려함 다듬기
    final = call_claude(
        f"""
아래는 논리적으로 점검된 고등학생 논설문이야.

[논설문]
{checked}

이 글의:
- 문장을 조금 더 자연스럽고 유려하게 다듬고
- 문단 간 연결을 부드럽게 이어주고
- 전체 톤은 '고등학생 수준의 정중한 논설문'으로 유지해줘.

최종 수정본만 보여줘.
"""
    )

    return final


def handle_mode_c_ideas(user_input: str) -> str:
    """
    C. 창의적 아이디어 모드:
    GPT: 20개 아이디어 → Gemini: 10개 선별 + 실현 가능성/확장 →
    Claude: 문장 정리
    """
    # 1) GPT로 아이디어 뽑기
    raw_ideas = call_gpt(
        f"""
너는 고등학생을 돕는 아이디어 브레인스토밍 도우미야.

[사용자 요청]
{user_input}

위 요청을 바탕으로, 서로 충분히 다른 아이디어를 최소 20개 bullet 목록으로 만들어줘.
각 아이디어는 한 줄 요약으로 작성해.
"""
    )

    # 2) Gemini로 선별 및 확장
    refined = call_gemini(
        f"""
너는 창의적이면서도 현실적인 아이디어를 골라주는 전문가야.

[사용자 요청]
{user_input}

[GPT가 만든 아이디어 20개]
{raw_ideas}

위 아이디어 중에서,
- 실현 가능성이 높고
- 교육적으로 의미 있고
- 어느 정도 창의적인 것

을 중심으로 10개만 고르는 동시에,
각 아이디어에 대해:
- 아이디어 제목
- 실현 가능성: 높음/보통/낮음
- 설명: 1~2문장
- 확장 아이디어: 1~2개 bullet

형식으로 정리해줘.
"""
    )

    # 3) Claude로 읽기 좋게 정리
    final = call_claude(
        f"""
아래는 고등학생을 위한 아이디어 목록이야.

{refined}

이 목록을:
- 문장만 조금 더 자연스럽게 다듬고
- 번호와 구성을 보기 좋게 정리해줘.
내용은 크게 바꾸지 말고 표현만 정리해줘.
"""
    )

    return final


def handle_mode_d_research(user_input: str) -> str:
    """
    D. 자료 조사 및 요약 모드:
    Perplexity: 웹 자료 수집/요약 → Gemini: 구조화/눈높이 정리
    """
    # 1) Perplexity로 자료 조사
    px_answer = call_perplexity(
        f"""
아래 주제에 대해 웹 자료를 조사해서,
핵심 내용과 중요한 포인트를 한국어로 요약해줘.
필요하면 간단한 출처도 함께 적어줘.

주제: {user_input}
"""
    )

    # 2) Gemini로 고등학생 기준 재정리
    final = call_gemini(
        f"""
아래는 웹에서 조사된 내용이야.

[조사 내용]
{px_answer}

이 내용을 고등학생이 이해하기 쉬운 수준으로,
- 소제목 3~5개
- 각 소제목 아래 bullet 2~4개

형태로 한국어로 정리해줘.
너무 어려운 전문 용어는 간단한 설명을 함께 붙여줘.
"""
    )

    return final


# =========================
# 9. 엔드포인트
# =========================

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    /chat 엔드포인트:
    입력 (mode, user_input)
      → 모드에 따른 처리
      → 여러 LLM 협업
      → 최종 answer 문자열 반환
    """
    global last_plan_for_a

    try:
        if req.mode == "A":
            text = req.user_input

            # "보고서 작성" 의도인지 판별
            want_report = is_report_request(text)

            if want_report:
                if last_plan_for_a is None:
                    answer = (
                        "[알림] 아직 1단계 탐구 방향/계획이 저장되어 있지 않습니다.\n"
                        "먼저 같은 주제로 A 모드에서 탐구 방향을 요청해 주세요.\n\n"
                        "예시:\n"
                        "A 모드에서\n"
                        '  "미세플라스틱과 수질 오염을 주제로 과학 탐구 방향을 설계해줘" 라고 먼저 요청한 뒤,\n'
                        '  그 다음에 "이제 위에서 추천한 2번 방향으로 탐구 보고서 작성해줘" 라고 요청하면 됩니다.'
                    )
                else:
                    answer = handle_mode_a_report(req.user_input, last_plan_for_a)
            else:
                # 탐구 방향/계획 생성 단계
                plan = handle_mode_a_plan(req.user_input)
                last_plan_for_a = plan
                answer = plan

        elif req.mode == "B":
            answer = handle_mode_b_essay(req.user_input)

        elif req.mode == "C":
            answer = handle_mode_c_ideas(req.user_input)

        else:  # "D"
            answer = handle_mode_d_research(req.user_input)

        return ChatResponse(answer=answer)

    except Exception as e:
        # 예외 발생 시, 일관된 형식으로 에러 메시지 반환
        # 프론트에서는 "서버 오류: ..." 형태로 표시됨
        raise HTTPException(
            status_code=500,
            detail=f"서버 내부 오류가 발생했습니다(AI 호출 또는 처리 중 문제): {e}",
        )
