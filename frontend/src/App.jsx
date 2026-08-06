import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  ArrowUp,
  CheckCircle2,
  Database,
  FileText,
  Plus,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'

const API_BASE =
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const examples = [
  'Why did the EGX30 rise and how did banking stocks perform?',
  'How much did Commercial International Bank (CIB) rise?',
  "What was Eastern Tobacco's quarterly net profit?",
  'Which companies announced dividends?',
]

function StatusPill({ health }) {
  const ready = health?.status === 'ok'

  return (
    <div className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white/80 px-3 py-1.5 text-xs font-medium text-slate-600 backdrop-blur">
      <span
        className={`h-2 w-2 rounded-full ${
          ready ? 'bg-emerald-500' : 'bg-amber-500'
        }`}
      />
      {ready ? 'Corpus and model ready' : 'Checking system'}
    </div>
  )
}

function ActionButton({ children, disabled = false, onClick, title }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 shadow-sm transition hover:border-indigo-300 hover:text-indigo-700 disabled:cursor-not-allowed disabled:opacity-40"
    >
      {children}
    </button>
  )
}

function SourceCard({ citation }) {
  const card = (
    <>
      <div className="mb-2 flex items-center justify-between gap-4">
        <span className="rounded-md bg-slate-100 px-2 py-1 text-[11px] font-bold tracking-wide text-slate-600">
          {citation.article_id}
        </span>
        <span className="text-xs text-slate-400">
          {Math.round(citation.relevance_score * 100)}% match
        </span>
      </div>

      <p className="font-semibold leading-snug text-slate-900 group-hover:text-indigo-700">
        {citation.title}
      </p>

      <p className="mt-2 text-xs text-slate-500">
        {citation.source || 'Local file'}
      </p>
    </>
  )

  if (!citation.url) {
    return (
      <div className="group block rounded-2xl border border-slate-200 bg-white p-4">
        {card}
      </div>
    )
  }

  return (
    <a
      href={citation.url}
      target="_blank"
      rel="noreferrer"
      className="group block rounded-2xl border border-slate-200 bg-white p-4 transition hover:-translate-y-0.5 hover:border-slate-300 hover:shadow-md"
    >
      {card}
    </a>
  )
}

function formatConversation(messages) {
  const body = messages
    .map((message) => {
      if (message.role === 'user') {
        return `USER\n${message.text}`
      }

      if (message.role === 'error') {
        return `ERROR\n${message.text}`
      }

      const sources = (message.citations || [])
        .map(
          (citation) =>
            `- ${citation.article_id}: ${citation.title} (${citation.source})`,
        )
        .join('\n')

      return [
        'ASSISTANT',
        message.answer || '',
        '',
        `Confidence: ${message.confidence || 'unknown'}`,
        `Latency: ${message.latency_ms || 0} ms`,
        `Sources: ${(message.citations || []).length}`,
        sources,
      ]
        .filter(Boolean)
        .join('\n')
    })
    .join('\n\n----------------------------------------\n\n')

  return `MarketBrief AI Conversation\nGenerated: ${new Date().toLocaleString()}\n\n${body}`
}

export default function App() {
  const [health, setHealth] = useState(null)
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState(() => crypto.randomUUID())

  const canSend = useMemo(
    () => question.trim().length >= 3 && !loading,
    [question, loading],
  )

  const hasConversation = messages.length > 0

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((response) => response.json())
      .then(setHealth)
      .catch(() => setHealth({ status: 'degraded' }))
  }, [])

  async function ask(text = question) {
    const clean = text.trim()

    if (clean.length < 3 || loading) {
      return
    }

    setQuestion('')
    setLoading(true)
    setMessages((current) => [...current, { role: 'user', text: clean }])

    try {
      const response = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: clean,
          session_id: sessionId,
        }),
      })

      const payload = await response.json()

      if (!response.ok) {
        throw new Error(payload.detail || 'Request failed')
      }

      setMessages((current) => [
        ...current,
        { role: 'assistant', ...payload },
      ])
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          role: 'error',
          text: error instanceof Error ? error.message : 'Request failed',
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  function handleNewChat() {
    if (loading) {
      return
    }

    setMessages([])
    setQuestion('')
    setSessionId(crypto.randomUUID())
  }

  function downloadConversation() {
    if (!hasConversation) {
      return
    }

    const blob = new Blob([formatConversation(messages)], {
      type: 'text/plain;charset=utf-8',
    })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')

    link.href = url
    link.download = `marketbrief-conversation-${new Date()
      .toISOString()
      .slice(0, 10)}.txt`

    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(url)
  }

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top_left,_#eef2ff,_transparent_38%),linear-gradient(180deg,#f8fafc_0%,#eef2f7_100%)] px-4 py-6 text-slate-900 sm:px-8">
      <div className="mx-auto max-w-6xl">
        <header className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
            <div className="grid h-11 w-11 place-items-center rounded-2xl bg-slate-950 text-white shadow-lg">
              <Activity size={21} />
            </div>

            <div>
              <p className="text-lg font-extrabold tracking-tight">
                MarketBrief AI
              </p>
              <p className="text-xs text-slate-500">
                AI-powered stock news research assistant
              </p>
            </div>
          </div>

          <StatusPill health={health} />
        </header>

        <section className="grid overflow-hidden rounded-[28px] border border-white/80 bg-white/75 shadow-panel backdrop-blur-xl lg:grid-cols-[1fr_310px]">
          <div className="flex min-h-[720px] flex-col border-b border-slate-200 lg:border-b-0 lg:border-r">
            <div className="border-b border-slate-200 px-6 py-5 sm:px-8">
              <div className="flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
                <div>
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-indigo-600">
                    <Sparkles size={14} />
                    AI Stock News Assistant
                  </div>

                  <h1 className="mt-2 text-2xl font-black tracking-tight sm:text-3xl">
                    Your AI-powered stock news research assistant
                  </h1>

                  <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-500">
                    Ask about market movements, company earnings, dividends,
                    IPOs, analyst opinions, regulations, financial figures, or
                    comparisons across the indexed articles.
                  </p>
                </div>

                <div className="flex flex-wrap gap-2">
                  <ActionButton
                    onClick={handleNewChat}
                    disabled={loading || !hasConversation}
                    title="Start a new conversation"
                  >
                    <Plus size={15} />
                    New Chat
                  </ActionButton>

                  <ActionButton
                    onClick={downloadConversation}
                    disabled={!hasConversation}
                    title="Download conversation as text"
                  >
                    <FileText size={15} />
                    Download TXT
                  </ActionButton>
                </div>
              </div>
            </div>

            <div className="flex-1 space-y-5 overflow-y-auto px-6 py-6 sm:px-8">
              {messages.length === 0 && (
                <div className="grid h-full place-items-center py-14">
                  <div className="max-w-2xl text-center">
                    <div className="mx-auto grid h-16 w-16 place-items-center rounded-2xl bg-indigo-50 text-indigo-600">
                      <Sparkles size={28} />
                    </div>

                    <h2 className="mt-5 text-xl font-extrabold">
                      I’m your AI-powered stock news assistant
                    </h2>

                    <p className="mx-auto mt-3 max-w-xl text-sm leading-7 text-slate-500">
                      I can summarize financial news, explain market movements,
                      compare companies, extract key figures, and answer
                      questions using only the indexed stock-market articles.
                    </p>

                    <div className="mt-6 flex flex-wrap justify-center gap-2">
                      {examples.map((example) => (
                        <button
                          type="button"
                          key={example}
                          onClick={() => ask(example)}
                          className="rounded-full border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-600 transition hover:border-indigo-300 hover:text-indigo-700"
                        >
                          {example}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {messages.map((message, index) => (
                <div
                  key={`${message.role}-${index}`}
                  className={
                    message.role === 'user'
                      ? 'ml-auto max-w-[82%]'
                      : 'max-w-[92%]'
                  }
                >
                  {message.role === 'user' && (
                    <div className="rounded-2xl rounded-br-md bg-slate-950 px-5 py-3.5 text-sm leading-6 text-white">
                      {message.text}
                    </div>
                  )}

                  {message.role === 'error' && (
                    <div className="rounded-2xl border border-red-200 bg-red-50 px-5 py-4 text-sm text-red-700">
                      {message.text}
                    </div>
                  )}

                  {message.role === 'assistant' && (
                    <div className="space-y-4">
                      <div className="whitespace-pre-wrap rounded-2xl rounded-bl-md border border-slate-200 bg-white px-5 py-4 text-sm leading-7 text-slate-700 shadow-sm">
                        {message.answer}
                      </div>

                      {message.safety_note && (
                        <div className="flex gap-3 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs leading-5 text-amber-900">
                          <ShieldCheck
                            className="mt-0.5 shrink-0"
                            size={16}
                          />
                          {message.safety_note}
                        </div>
                      )}

                      <div className="flex flex-wrap gap-2 text-[11px] font-medium text-slate-500">
                        <span className="rounded-full bg-slate-100 px-2.5 py-1">
                          Confidence: {message.confidence}
                        </span>
                        <span className="rounded-full bg-slate-100 px-2.5 py-1">
                          {message.latency_ms} ms
                        </span>
                        <span className="rounded-full bg-slate-100 px-2.5 py-1">
                          {(message.citations || []).length} source(s)
                        </span>
                        <span className="rounded-full bg-slate-100 px-2.5 py-1">
                          {message.usage?.input_tokens || 0} input tokens
                        </span>
                        <span className="rounded-full bg-slate-100 px-2.5 py-1">
                          {message.usage?.output_tokens || 0} output tokens
                        </span>
                      </div>

                      {(message.citations || []).length > 0 && (
                        <div className="grid gap-3 sm:grid-cols-2">
                          {message.citations.map((citation) => (
                            <SourceCard
                              key={citation.article_id}
                              citation={citation}
                            />
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}

              {loading && (
                <div className="inline-flex items-center gap-2 rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-500">
                  <span className="h-2 w-2 animate-pulse rounded-full bg-indigo-500" />
                  Retrieving and validating evidence…
                </div>
              )}
            </div>

            <div className="border-t border-slate-200 bg-white/80 p-4 sm:p-6">
              <div className="flex items-end gap-3 rounded-2xl border border-slate-200 bg-white p-2 shadow-sm focus-within:border-indigo-400 focus-within:ring-4 focus-within:ring-indigo-50">
                <textarea
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && !event.shiftKey) {
                      event.preventDefault()
                      ask()
                    }
                  }}
                  rows={2}
                  placeholder="Ask about earnings, dividends, market movements, or compare articles…"
                  className="min-h-[52px] flex-1 resize-none border-0 px-3 py-2 text-sm outline-none placeholder:text-slate-400"
                />

                <button
                  type="button"
                  onClick={() => ask()}
                  disabled={!canSend}
                  aria-label="Send question"
                  className="grid h-11 w-11 shrink-0 place-items-center rounded-xl bg-indigo-600 text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-200"
                >
                  <ArrowUp size={18} />
                </button>
              </div>

              <p className="mt-2 text-center text-[11px] text-slate-400">
                Research summary only — not personalized investment advice.
              </p>
            </div>
          </div>

          <aside className="bg-slate-50/70 p-6">
            <h2 className="text-sm font-bold">
              Why you can trust this answer
            </h2>

            <div className="mt-5 space-y-5">
              {[
                [
                  Database,
                  'Hybrid retrieval',
                  'Dense semantic search plus BM25 for tickers, numbers, and exact terms.',
                ],
                [
                  CheckCircle2,
                  'Citation validation',
                  'The API verifies that every cited article was retrieved.',
                ],
                [
                  ShieldCheck,
                  'Financial guardrail',
                  'Recommendation requests are converted into neutral research summaries.',
                ],
              ].map(([Icon, title, copy]) => (
                <div key={title} className="flex gap-3">
                  <div className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-white text-indigo-600 shadow-sm">
                    <Icon size={17} />
                  </div>

                  <div>
                    <p className="text-sm font-semibold">{title}</p>
                    <p className="mt-1 text-xs leading-5 text-slate-500">
                      {copy}
                    </p>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-8 rounded-2xl bg-slate-950 p-5 text-white">
              <p className="text-xs font-semibold uppercase tracking-[0.14em] text-indigo-300">
                Audit-ready
              </p>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                Each response stores retrieval scores, sources, latency, token
                usage, estimated cost, corpus version, and validation flags.
              </p>
            </div>
          </aside>
        </section>
      </div>
    </main>
  )
}