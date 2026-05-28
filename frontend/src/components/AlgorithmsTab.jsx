// ── Static algorithm metadata ─────────────────────────────────────────────────
const ALGO_META = {
  DQN: {
    fullName:     'Deep Q-Network',
    type:         'Value-Based · Off-Policy',
    typeColor:    '#2962ff',
    accent:       '#5c8df6',
    description:
      'Learns a Q-table approximated by a neural network. An experience replay buffer and a frozen ' +
      'target network stabilise training — the network gradually learns which actions maximise ' +
      'future discounted reward.',
    strengths:    ['Stable with experience replay', 'Clear buy/sell signal via Q-gap', 'Fast inference'],
    weaknesses:   ['Overestimates Q-values', 'Needs tuned ε-schedule'],
    architecture: '3-layer MLP (obs→128→64→3) · ReLU · ε-greedy · Adam',
    bestFor:      'Trending markets with clear directional momentum',
    trainNote:    '5 000 episodes · random-start · continuous P&L reward',
    recommended:  { epsilon: 0.02, risk_aversion: 0.3, position_size: 1.0, stop_loss: 15, take_profit: 25 },
  },
  DDQN: {
    fullName:     'Double Deep Q-Network',
    type:         'Value-Based · Off-Policy',
    typeColor:    '#1a6bd4',
    accent:       '#4da6ff',
    description:
      'Fixes the systematic overestimation bias in DQN by decoupling action selection ' +
      '(online network) from action evaluation (target network). Produces more conservative, ' +
      'accurate value estimates — especially valuable in volatile stock environments.',
    strengths:    ['Accurate Q-values', 'Less overfit to spike rewards', 'More conservative trades'],
    weaknesses:   ['Slower initial learning', 'Same replay-buffer dependency as DQN'],
    architecture: 'Dual MLP (online + target) · Decoupled argmax · Polyak-averaging target sync',
    bestFor:      'Mixed-trend markets where DQN overbets',
    trainNote:    '5 000 episodes · same env as DQN · soft target update τ=0.005',
    recommended:  { epsilon: 0.02, risk_aversion: 0.35, position_size: 1.0, stop_loss: 12, take_profit: 20 },
  },
  A2C: {
    fullName:     'Advantage Actor-Critic',
    type:         'Policy Gradient · On-Policy',
    typeColor:    '#9c27b0',
    accent:       '#ce93d8',
    description:
      'Simultaneously learns a policy π(a|s) — the actor — and a value function V(s) — the critic. ' +
      'The advantage A(s,a) = r + γV(s′) − V(s) tells the actor how much better an action was ' +
      'versus the average. Entropy bonus encourages exploration.',
    strengths:    ['Direct policy learning', 'Built-in exploration via entropy', 'Continuous action semantics'],
    weaknesses:   ['On-policy: less sample efficient', 'Sensitive to learning rate'],
    architecture: 'Shared trunk (obs→128→64) → actor head + critic head · Advantage estimator',
    bestFor:      'Volatile markets where value-based methods oscillate',
    trainNote:    '5 000 episodes · on-policy rollouts · entropy coef 0.01',
    recommended:  { temperature: 0.8, risk_aversion: 0.5, position_size: 0.8, stop_loss: 20, take_profit: 30 },
  },
  PPO: {
    fullName:     'Proximal Policy Optimisation',
    type:         'Policy Gradient · On-Policy',
    typeColor:    '#e65100',
    accent:       '#ff9800',
    description:
      'Extends A2C with a clipped surrogate objective that prevents the policy update ratio from ' +
      'deviating too far from the old policy. This "trust region" constraint makes training ' +
      'dramatically more stable — PPO is the industry default for RL in finance.',
    strengths:    ['State-of-the-art stability', 'Robust to hyperparameter choices', 'Best generalisation'],
    weaknesses:   ['On-policy (needs many steps)', 'Clip threshold requires tuning'],
    architecture: 'Actor-Critic · Clipped ratio clip ε=0.2 · GAE λ=0.95 · mini-batch updates',
    bestFor:      'General deployment — most reliable in unseen regimes',
    trainNote:    '5 000 episodes · clip=0.2 · 4 PPO epochs per rollout',
    recommended:  { temperature: 1.0, risk_aversion: 0.4, position_size: 0.9, stop_loss: 18, take_profit: 35 },
  },
  Random: {
    fullName:     'Random Baseline',
    type:         'Baseline · Stochastic',
    typeColor:    '#4c525e',
    accent:       '#787b86',
    description:
      'Selects buy, sell, or hold uniformly at random with no learning. ' +
      'Exists solely as a sanity check — every trained agent must beat it consistently. ' +
      'If a trained agent loses to Random, the training is broken.',
    strengths:    ['Zero training time', 'Zero overfitting'],
    weaknesses:   ['No intelligence', 'Expected return = 0 minus commission'],
    architecture: 'Uniform random over {Buy, Sell, Hold}',
    bestFor:      'Baseline comparison only — never deploy',
    trainNote:    'No training required',
    recommended:  null,
  },
}

// ── helpers ───────────────────────────────────────────────────────────────────

function MetricPill({ label, value, color }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      background: '#1a1e27', borderRadius: 6, padding: '7px 10px', flex: 1,
    }}>
      <span style={{ fontSize: 13, fontWeight: 700, color: color ?? '#d1d4dc',
        fontVariantNumeric: 'tabular-nums' }}>
        {value ?? '—'}
      </span>
      <span style={{ fontSize: 9, color: '#4c525e', textTransform: 'uppercase',
        letterSpacing: '0.06em', marginTop: 2 }}>
        {label}
      </span>
    </div>
  )
}

function Tag({ children, color }) {
  return (
    <span style={{
      fontSize: 9, fontWeight: 700, padding: '2px 6px', borderRadius: 3,
      background: color + '22', border: `1px solid ${color}55`, color,
      letterSpacing: '0.04em', textTransform: 'uppercase',
    }}>
      {children}
    </span>
  )
}

function AlgoCard({ algoId, meta, result, loaded, spyReturn, onApply }) {
  const ret    = result?.total_return_pct
  const sharpe = result?.sharpe_ratio
  const maxDD  = result?.max_drawdown
  const wr     = result?.win_rate
  const trades = result?.n_trades ?? result?.total_trades

  const retColor    = ret    == null ? '#787b86' : ret    >= 0   ? '#26a69a' : '#ef5350'
  const sharpeColor = sharpe == null ? '#787b86' : sharpe >= 1   ? '#26a69a' : sharpe >= 0 ? '#d1d4dc' : '#ef5350'
  const wrColor     = wr     == null ? '#787b86' : wr     >= 50  ? '#26a69a' : '#ef5350'

  const vsSpy = ret != null && spyReturn != null
    ? Math.round((ret - spyReturn) * 100) / 100
    : null
  const vsSpyColor = vsSpy == null ? '#787b86' : vsSpy >= 0 ? '#26a69a' : '#ef5350'

  const isRandom = algoId === 'Random'

  return (
    <div style={{
      background: '#1e2329',
      border: `1px solid #2a2e39`,
      borderTop: `3px solid ${meta.accent}`,
      borderRadius: 8,
      padding: '16px',
      display: 'flex',
      flexDirection: 'column',
      gap: 12,
    }}>

      {/* ── Header ── */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span style={{ fontSize: 16, fontWeight: 800, color: '#d1d4dc' }}>{algoId}</span>
            <Tag color={meta.typeColor}>{meta.type.split(' · ')[0]}</Tag>
            {!isRandom && (
              <span style={{
                fontSize: 9, fontWeight: 700, padding: '2px 6px', borderRadius: 3,
                background: loaded ? 'rgba(38,166,154,0.15)' : 'rgba(239,83,80,0.12)',
                border: `1px solid ${loaded ? '#26a69a55' : '#ef535055'}`,
                color: loaded ? '#26a69a' : '#ef5350',
              }}>
                {loaded ? '✓ Trained' : '✗ Not Loaded'}
              </span>
            )}
          </div>
          <div style={{ fontSize: 11, color: '#787b86' }}>{meta.fullName}</div>
        </div>
      </div>

      {/* ── Description ── */}
      <p style={{ fontSize: 11, color: '#787b86', lineHeight: 1.6, margin: 0 }}>
        {meta.description}
      </p>

      {/* ── Performance metrics ── */}
      <div>
        <div style={{ fontSize: 9, fontWeight: 700, color: '#4c525e',
          textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>
          Live Performance · Test Period
        </div>
        <div style={{ display: 'flex', gap: 4 }}>
          <MetricPill label="Return"
            value={ret != null ? `${ret >= 0 ? '+' : ''}${ret.toFixed(2)}%` : '—'}
            color={retColor} />
          <MetricPill label="Sharpe"
            value={sharpe != null ? sharpe.toFixed(3) : '—'}
            color={sharpeColor} />
          <MetricPill label="Max DD"
            value={maxDD != null ? `${(maxDD * 100).toFixed(1)}%` : '—'}
            color="#ef5350" />
          <MetricPill label="Win Rate"
            value={wr != null ? `${wr.toFixed(0)}%` : '—'}
            color={wrColor} />
          <MetricPill label="vs S&P"
            value={vsSpy != null ? `${vsSpy >= 0 ? '+' : ''}${vsSpy.toFixed(2)}%` : '—'}
            color={vsSpyColor} />
        </div>
      </div>

      {/* ── Architecture + strengths ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        <div>
          <div style={{ fontSize: 9, fontWeight: 700, color: '#4c525e',
            textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 5 }}>
            Architecture
          </div>
          <div style={{ fontSize: 10, color: '#787b86', lineHeight: 1.6 }}>
            {meta.architecture}
          </div>
          <div style={{ fontSize: 10, color: '#4c525e', marginTop: 4, fontStyle: 'italic' }}>
            {meta.trainNote}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 9, fontWeight: 700, color: '#4c525e',
            textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 5 }}>
            Strengths / Weaknesses
          </div>
          {meta.strengths.map(s => (
            <div key={s} style={{ fontSize: 10, color: '#26a69a', lineHeight: 1.7 }}>+ {s}</div>
          ))}
          {meta.weaknesses.map(w => (
            <div key={w} style={{ fontSize: 10, color: '#ef5350', lineHeight: 1.7 }}>− {w}</div>
          ))}
        </div>
      </div>

      {/* ── Best for ── */}
      <div style={{
        background: '#2a2e39', borderLeft: `3px solid ${meta.accent}`,
        borderRadius: 4, padding: '6px 10px',
        fontSize: 10, color: '#787b86',
      }}>
        <span style={{ color: meta.accent, fontWeight: 700 }}>Best for: </span>
        {meta.bestFor}
      </div>

      {/* ── Recommended settings + Apply button ── */}
      {meta.recommended && !isRandom && (
        <div>
          <div style={{ fontSize: 9, fontWeight: 700, color: '#4c525e',
            textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>
            Recommended Inference Settings
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginBottom: 10 }}>
            {meta.recommended.epsilon   != null && <Tag color="#5c8df6">ε = {meta.recommended.epsilon}</Tag>}
            {meta.recommended.temperature != null && <Tag color="#5c8df6">T = {meta.recommended.temperature}</Tag>}
            <Tag color="#787b86">Risk = {meta.recommended.risk_aversion}</Tag>
            <Tag color="#787b86">Size = {Math.round(meta.recommended.position_size * 100)}%</Tag>
            <Tag color="#26a69a">SL = −{meta.recommended.stop_loss}%</Tag>
            <Tag color="#f5c842">TP = +{meta.recommended.take_profit}%</Tag>
          </div>
          <button
            onClick={() => onApply(algoId, meta.recommended)}
            style={{
              width: '100%', padding: '10px 0', borderRadius: 5, border: 'none',
              background: `linear-gradient(135deg, ${meta.accent} 0%, ${meta.typeColor} 100%)`,
              color: '#fff', fontSize: 12, fontWeight: 700,
              cursor: 'pointer', letterSpacing: '0.02em',
              transition: 'opacity 0.15s',
            }}
            onMouseEnter={e => e.currentTarget.style.opacity = '0.85'}
            onMouseLeave={e => e.currentTarget.style.opacity = '1'}
          >
            Apply Recommended Settings &amp; Run Backtest
          </button>
        </div>
      )}

      {isRandom && (
        <div style={{
          textAlign: 'center', fontSize: 11, color: '#4c525e',
          padding: '8px 0', fontStyle: 'italic',
        }}>
          No recommended settings — baseline only
        </div>
      )}
    </div>
  )
}

// ── main export ───────────────────────────────────────────────────────────────

export default function AlgorithmsTab({ results, models, spyBenchmark, onApply }) {
  const resultMap = Object.fromEntries((results ?? []).map(r => [r.algo, r]))
  const spyReturn = spyBenchmark?.total_return_pct ?? null

  const ORDER = ['DQN', 'DDQN', 'A2C', 'PPO', 'Random']

  return (
    <div style={{ padding: '20px', overflowY: 'auto', height: '100%', boxSizing: 'border-box' }}>

      {/* Page header */}
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 14, fontWeight: 800, color: '#d1d4dc', letterSpacing: '-0.02em', marginBottom: 4 }}>
          Trained Algorithms
        </div>
        <div style={{ fontSize: 11, color: '#4c525e', lineHeight: 1.6 }}>
          Four reinforcement learning strategies, each with a distinct approach to learning optimal
          trading policy. Performance shown on the held-out test period.
          Click <span style={{ color: '#d1d4dc' }}>Apply Recommended Settings</span> to pre-configure
          the backtest with the best known inference parameters for that algorithm.
        </div>
      </div>

      {/* Algorithm cards — 2-column grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(420px, 1fr))',
        gap: 16,
      }}>
        {ORDER.map(id => {
          const meta   = ALGO_META[id]
          const result = resultMap[id]
          const loaded = id === 'Random' ? true : (models?.[id.toLowerCase()] ?? false)
          return (
            <AlgoCard
              key={id}
              algoId={id}
              meta={meta}
              result={result}
              loaded={loaded}
              spyReturn={spyReturn}
              onApply={onApply}
            />
          )
        })}
      </div>

      {/* Footer note */}
      <div style={{
        marginTop: 20, padding: '10px 14px',
        background: 'rgba(41,98,255,0.06)',
        border: '1px solid rgba(41,98,255,0.18)',
        borderRadius: 6, fontSize: 10, color: '#787b86', lineHeight: 1.7,
      }}>
        <span style={{ color: '#5c8df6', fontWeight: 700 }}>ℹ</span>
        {'  '}Recommended settings are starting points derived from algorithm characteristics — not
        guaranteed optima. Use the{' '}
        <span style={{ color: '#d1d4dc' }}>Backtest</span> tab to experiment with stop-loss and
        take-profit levels, or{' '}
        <span style={{ color: '#d1d4dc' }}>Run All 5 Algos &amp; Compare</span> to see them side-by-side
        under the same conditions.
      </div>
    </div>
  )
}
