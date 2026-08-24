import React,{useMemo,useState} from 'react';
import {createRoot} from 'react-dom/client';
import {Upload,FileAudio,Users,Clock3,Mic2,Search,Download,PlayCircle,AlertCircle} from 'lucide-react';
import './styles.css';

const API_URL='https://per-actai.onrender.com';

const fmt=(seconds=0)=>{
 const s=Math.max(0,Math.floor(seconds));
 const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=s%60;
 return h>0?`${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`:`${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`;
};

function App(){
 const [file,setFile]=useState(null),[tab,setTab]=useState('resumen'),[query,setQuery]=useState(''),[data,setData]=useState(null),[busy,setBusy]=useState(false),[error,setError]=useState('');

 const analyze=async()=>{
  if(!file)return;
  setBusy(true);setError('');
  try{
   const fd=new FormData();fd.append('file',file);
   const res=await fetch(`${API_URL}/analyze`,{method:'POST',body:fd});
   const body=await res.json().catch(()=>({}));
   if(!res.ok) throw new Error(body.detail||`Error HTTP ${res.status}`);
   const transcript=(body.segments||[]).map(x=>({
    time:fmt(x.start),speaker:x.speaker||'SPEAKER_00',text:x.text||'',start:x.start||0,end:x.end||0
   }));
   const speakers=[...new Set(transcript.map(x=>x.speaker))].map(id=>({id,name:id,time:'—',pct:'—',turns:transcript.filter(x=>x.speaker===id).length,topics:'Diarización pendiente'}));
   setData({
    title:body.filename||file.name,
    duration:fmt(body.duration||0),
    language:body.language||'es',
    languageProbability:body.language_probability,
    transcript,
    fullTranscript:body.transcript||'',
    speakers,
    summary:'La transcripción real del audio ya fue generada por Whisper. En esta versión todavía no se genera el resumen automático ni la separación real de hablantes.',
    topics:['Transcripción real completada','Resumen automático pendiente','Diarización de hablantes pendiente'],
    agreements:[]
   });
   setTab('transcripcion');
  }catch(e){
   setError(e.message||'No se pudo procesar el audio.');
  }finally{setBusy(false)}
 };

 const visible=useMemo(()=>!data?[]:data.transcript.filter(x=>(x.text+' '+x.speaker).toLowerCase().includes(query.toLowerCase())),[data,query]);

 return <div className="app"><header><div><div className="brand"><Mic2 size={22}/> ActaAI</div><p>Convierte conversaciones en información útil.</p></div><span className="badge">MVP · v0.2</span></header>
 <main>{!data?<section className="hero"><div className="heroText"><span className="eyebrow">TRANSCRIPCIÓN REAL CON WHISPER</span><h1>Sube un audio.<br/>Obtén la transcripción.</h1><p>El archivo se envía a tu backend en Render y se procesa con faster-whisper.</p></div><div className="uploadCard"><label className="drop"><Upload size={35}/><strong>{file?file.name:'Selecciona un audio'}</strong><span>MP3, WAV, M4A, MP4, OGG o FLAC</span><input type="file" accept="audio/*,video/mp4" onChange={e=>{setFile(e.target.files?.[0]||null);setError('')}}/></label>{file&&<div className="fileRow"><FileAudio/><div><b>{file.name}</b><small>{(file.size/1024/1024).toFixed(1)} MB</small></div></div>}<button className="primary" disabled={!file||busy} onClick={analyze}>{busy?'Procesando en Render…':'Analizar audio real'}</button>{busy&&<small className="hint">La primera solicitud puede tardar si Render estaba en reposo. El procesamiento de Whisper también puede demorar varios minutos.</small>}{error&&<div className="errorBox"><AlertCircle size={18}/><span>{error}</span></div>}<small className="hint">Backend: {API_URL}</small></div></section>:
 <section><div className="topline"><div><button className="back" onClick={()=>{setData(null);setQuery('')}}>← Nuevo audio</button><h2>{data.title}</h2><p>{data.duration} · idioma {data.language} · {data.speakers.length} hablante provisional</p></div><button className="export" onClick={()=>{const blob=new Blob([data.fullTranscript],{type:'text/plain;charset=utf-8'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=`${data.title}-transcripcion.txt`;a.click();URL.revokeObjectURL(url)}}><Download size={16}/> Exportar TXT</button></div>
 <nav>{[['resumen','Resumen'],['transcripcion','Transcripción'],['participantes','Participantes'],['tareas','Acuerdos y tareas']].map(([k,l])=><button className={tab===k?'active':''} onClick={()=>setTab(k)} key={k}>{l}</button>)}</nav>
 {tab==='resumen'&&<div className="grid"><div className="card wide"><h3>Estado del análisis</h3><p className="summary">{data.summary}</p></div><div className="card"><h3>Pipeline</h3>{data.topics.map((t,i)=><div className="topic" key={t}><span>{i+1}</span>{t}</div>)}</div><div className="card"><h3>Datos técnicos</h3><p><b>Duración:</b> {data.duration}</p><p><b>Idioma:</b> {data.language}</p><p><b>Segmentos:</b> {data.transcript.length}</p></div></div>}
 {tab==='transcripcion'&&<div className="card"><div className="search"><Search/><input placeholder="Buscar en la transcripción…" value={query} onChange={e=>setQuery(e.target.value)}/></div><div className="transcript">{visible.map((x,i)=><div className="line" key={`${x.start}-${i}`}><button className="time"><PlayCircle size={14}/>{x.time}</button><div><b>{x.speaker}</b><p>{x.text}</p></div></div>)}</div></div>}
 {tab==='participantes'&&<div className="people">{data.speakers.map(s=><div className="person card" key={s.id}><div className="avatar">{s.name[0]}</div><h3>{s.name}</h3><span>{s.id}</span><div className="stats"><div><strong>{s.turns}</strong><small>segmentos</small></div><div><strong>—</strong><small>participación</small></div><div><strong>—</strong><small>tiempo</small></div></div><p><b>Estado:</b> diarización pendiente.</p></div>)}</div>}
 {tab==='tareas'&&<div className="card"><h3>Acuerdos y tareas</h3><p>Esta etapa todavía no está conectada. El siguiente paso es generar resumen, acuerdos, responsables y tareas desde la transcripción real.</p></div>}
 </section>}</main></div>
}
createRoot(document.getElementById('root')).render(<App/>);
