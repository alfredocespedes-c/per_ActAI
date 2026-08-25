// ActaAI quick enhancements: compressed live recording + paste-text analysis.
// Loaded before React so MediaRecorder defaults are optimized for speech.
(() => {
  const NativeMediaRecorder = window.MediaRecorder;
  if (NativeMediaRecorder) {
    function SpeechMediaRecorder(stream, options = {}) {
      const opts = { ...options };
      if (!opts.audioBitsPerSecond) opts.audioBitsPerSecond = 40000;
      return new NativeMediaRecorder(stream, opts);
    }
    SpeechMediaRecorder.prototype = NativeMediaRecorder.prototype;
    SpeechMediaRecorder.isTypeSupported = NativeMediaRecorder.isTypeSupported.bind(NativeMediaRecorder);
    window.MediaRecorder = SpeechMediaRecorder;
  }

  const STOP = new Set('para como pero porque cuando donde desde hasta sobre entre este esta estos estas esto eso esa ese aqui ahi muy mas menos tambien entonces bueno bien vamos tiene tener hacer hace hecho hay que del las los una uno unos unas por con sin de la el en un al se es lo le me te nos su sus mi mis ya si'.split(' '));
  const words = t => (t.toLowerCase().match(/[a-záéíóúñü]{4,}/g) || []).filter(w => !STOP.has(w));
  const sentences = t => (t.match(/[^.!?\n]+[.!?]?/g) || []).map(s => s.trim()).filter(Boolean);
  const analyzeText = text => {
    const freq = {};
    words(text).forEach(w => freq[w] = (freq[w] || 0) + 1);
    const topics = Object.entries(freq).sort((a,b)=>b[1]-a[1]).slice(0,6).map(([w])=>w[0].toUpperCase()+w.slice(1));
    const scored = sentences(text).map((s,i)=>({s,i,score:words(s).reduce((n,w)=>n+(freq[w]||0),0)/Math.max(words(s).length,1)}));
    const summary = scored.sort((a,b)=>b.score-a.score).slice(0,5).sort((a,b)=>a.i-b.i).map(x=>x.s).join(' ');
    const taskRe = /\b(hay que|tenemos que|tengo que|debe|debemos|deberíamos|necesitamos|voy a|me encargo|acordamos|queda pendiente|vamos a)\b/i;
    const tasks = sentences(text).filter(s=>taskRe.test(s)).slice(0,12);
    return {summary: summary || text.slice(0,900), topics, tasks};
  };
  const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

  function install() {
    const sw = document.querySelector('.modeSwitch');
    if (!sw || sw.querySelector('[data-text-mode]')) return;
    const btn = document.createElement('button');
    btn.dataset.textMode = '1';
    btn.innerHTML = '✍️ Pegar texto';
    sw.appendChild(btn);
    btn.addEventListener('click', () => {
      sw.querySelectorAll('button').forEach(b=>b.classList.remove('selected'));
      btn.classList.add('selected');
      const card = sw.closest('.uploadCard');
      [...card.children].forEach(el=>{if(el!==sw) el.style.display='none'});
      let panel = card.querySelector('.pasteTextPanel');
      if (!panel) {
        panel = document.createElement('div'); panel.className='pasteTextPanel';
        panel.innerHTML = `<div style="display:grid;gap:12px;margin-top:18px"><textarea data-paste-input placeholder="Pega aquí una transcripción, notas o texto de una reunión…" style="min-height:260px;width:100%;box-sizing:border-box;padding:16px;border:1px solid #d8dde6;border-radius:14px;font:inherit;resize:vertical"></textarea><button data-analyze-text class="primary">Analizar texto</button><div data-text-error style="display:none;color:#b42318"></div><div data-text-result></div></div>`;
        card.appendChild(panel);
        panel.querySelector('[data-analyze-text]').onclick=()=>{
          const text=panel.querySelector('[data-paste-input]').value.trim(), err=panel.querySelector('[data-text-error]'), out=panel.querySelector('[data-text-result]');
          if(text.length<20){err.textContent='Pega un texto de al menos 20 caracteres.';err.style.display='block';return} err.style.display='none';
          const r=analyzeText(text);
          out.innerHTML=`<div style="margin-top:8px;padding:18px;border:1px solid #e4e7ec;border-radius:14px"><h3>Resumen</h3><p>${esc(r.summary)}</p><h3>Temas</h3><p>${r.topics.map(esc).join(' · ')||'Sin temas detectados'}</p><h3>Acuerdos y tareas</h3>${r.tasks.length?'<ul>'+r.tasks.map(t=>'<li>'+esc(t)+'</li>').join('')+'</ul>':'<p>No se detectaron compromisos explícitos.</p>'}<button data-copy-text class="secondaryBtn" style="margin-top:8px">Copiar análisis</button></div>`;
          out.querySelector('[data-copy-text]').onclick=()=>navigator.clipboard?.writeText(`RESUMEN\n${r.summary}\n\nTEMAS\n${r.topics.join(', ')}\n\nACUERDOS Y TAREAS\n${r.tasks.join('\n')}\n\nTEXTO ORIGINAL\n${text}`);
        };
      }
      panel.style.display='block';
    });
    [...sw.querySelectorAll('button')].filter(b=>b!==btn).forEach(b=>b.addEventListener('click',()=>{const p=sw.closest('.uploadCard')?.querySelector('.pasteTextPanel');if(p)p.style.display='none'}));
  }
  new MutationObserver(install).observe(document.documentElement,{childList:true,subtree:true});
  document.addEventListener('DOMContentLoaded',install);
})();
