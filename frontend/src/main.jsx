import React,{useMemo,useRef,useState} from 'react';
import {createRoot} from 'react-dom/client';
import {Upload,FileAudio,Users,Clock3,Mic2,Search,Download,PlayCircle,AlertCircle,Mail,Edit3,CheckCircle2,Pause,Square,Copy,Radio,RotateCcw} from 'lucide-react';
import './styles.css';

const API_URL='https://per-actai.onrender.com';
const fmt=(seconds=0)=>{const s=Math.max(0,Math.floor(seconds));const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=s%60;return h>0?`${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`:`${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`};

function speakerStats(transcript){
 const totals={};let all=0;
 transcript.forEach(x=>{const d=Math.max(0,(x.end||0)-(x.start||0));all+=d;if(!totals[x.speaker])totals[x.speaker]={id:x.speaker,name:x.speaker,seconds:0,turns:0};totals[x.speaker].seconds+=d;totals[x.speaker].turns+=1});
 return Object.values(totals).map(s=>({...s,time:fmt(s.seconds),pct:all?Math.round(s.seconds/all*100):0}));
}

function fullActa(data){
 const lines=[`ACTA DE ${data.title}`,`Duración: ${data.duration}`,`Participantes: ${data.speakers.length}`,'','RESUMEN',data.summary,'','TEMAS TRATADOS',...data.topics.map((t,i)=>`${i+1}. ${t}`),'','ACUERDOS Y TAREAS',...(data.agreements.length?data.agreements.map((a,i)=>`${i+1}. ${a.text}\n   Responsable: ${a.owner} · ${a.due}`):['No se detectaron compromisos explícitos.']),'','TRANSCRIPCIÓN',...data.transcript.map(x=>`[${x.time}] ${x.speaker}: ${x.text}`)];
 return lines.join('\n');
}

function downloadText(text,name){
 const blob=new Blob([text],{type:'text/plain;charset=utf-8'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;a.click();URL.revokeObjectURL(url);
}

function App(){
 const [file,setFile]=useState(null),[tab,setTab]=useState('resumen'),[query,setQuery]=useState(''),[data,setData]=useState(null),[busy,setBusy]=useState(false),[error,setError]=useState('');
 const [mode,setMode]=useState('upload');
 const [liveState,setLiveState]=useState('idle');
 const [liveFinal,setLiveFinal]=useState('');
 const [liveInterim,setLiveInterim]=useState('');
 const [liveError,setLiveError]=useState('');
 const recognitionRef=useRef(null);
 const liveSupported=typeof window!=='undefined'&&Boolean(window.SpeechRecognition||window.webkitSpeechRecognition);

 const analyze=async()=>{if(!file)return;setBusy(true);setError('');try{const fd=new FormData();fd.append('file',file);const res=await fetch(`${API_URL}/analyze`,{method:'POST',body:fd});const body=await res.json().catch(()=>({}));if(!res.ok)throw new Error(body.detail||'No se pudo analizar el audio');const transcript=(body.segments||[]).map(x=>({time:fmt(x.start),speaker:x.speaker||'Hablante 1',text:x.text||'',start:x.start||0,end:x.end||0}));const speakers=speakerStats(transcript);setData({title:body.filename||file.name,duration:fmt(body.duration||0),language:body.language||'es',transcript,fullTranscript:body.transcript||'',speakers,summary:body.summary||'No se pudo generar un resumen.',topics:body.topics||[],agreements:body.agreements||[]});setTab('resumen')}catch(e){setError(e.message||'No se pudo procesar el audio.')}finally{setBusy(false)}};
 const visible=useMemo(()=>!data?[]:data.transcript.filter(x=>(x.text+' '+x.speaker).toLowerCase().includes(query.toLowerCase())),[data,query]);
 const renameSpeaker=(oldName)=>{const name=window.prompt('Nombre del participante',oldName);if(!name||name.trim()===oldName)return;const clean=name.trim();const transcript=data.transcript.map(x=>x.speaker===oldName?{...x,speaker:clean}:x);const agreements=data.agreements.map(a=>a.owner===oldName?{...a,owner:clean}:a);setData({...data,transcript,agreements,speakers:speakerStats(transcript)})};
 const exportTxt=()=>downloadText(fullActa(data),`${data.title}-acta.txt`);
 const sendEmail=(type)=>{const subject=encodeURIComponent(`${type==='transcript'?'Transcripción':'Acta'} - ${data.title}`);let body=type==='transcript'?data.transcript.map(x=>`[${x.time}] ${x.speaker}: ${x.text}`).join('\n'):fullActa(data);if(body.length>14000)body=body.slice(0,14000)+'\n\n[Contenido abreviado por límite del cliente de correo. Puedes adjuntar el archivo exportado desde ActaAI.]';window.location.href=`mailto:?subject=${subject}&body=${encodeURIComponent(body)}`};

 const createRecognition=()=>{
  const SpeechRecognition=window.SpeechRecognition||window.webkitSpeechRecognition;
  if(!SpeechRecognition)return null;
  const recognition=new SpeechRecognition();
  recognition.lang='es-CL';
  recognition.continuous=true;
  recognition.interimResults=true;
  recognition.maxAlternatives=1;
  recognition.onresult=(event)=>{let finalChunk='',interim='';for(let i=event.resultIndex;i<event.results.length;i++){const text=event.results[i][0].transcript;if(event.results[i].isFinal)finalChunk+=text+' ';else interim+=text}if(finalChunk)setLiveFinal(prev=>(prev+(prev&&!prev.endsWith(' ')?' ':'')+finalChunk).trimStart());setLiveInterim(interim)};
  recognition.onerror=(event)=>{if(event.error==='aborted'||event.error==='no-speech')return;setLiveError(event.error==='not-allowed'?'El navegador bloqueó el micrófono. Autoriza su uso e inténtalo de nuevo.':`Error de reconocimiento: ${event.error}`);setLiveState('idle')};
  recognition.onend=()=>{if(recognitionRef.current===recognition&&liveState==='listening'){try{recognition.start()}catch{}}};
  recognitionRef.current=recognition;
  return recognition;
 };
 const startLive=()=>{setLiveError('');if(!liveSupported){setLiveError('La transcripción en vivo necesita un navegador compatible con Web Speech API, como Chrome o Edge.');return}const recognition=createRecognition();if(!recognition)return;try{recognition.start();setLiveState('listening')}catch(e){setLiveError('No se pudo iniciar el micrófono. Revisa los permisos del navegador.')}};
 const pauseLive=()=>{const recognition=recognitionRef.current;if(recognition){recognitionRef.current=null;try{recognition.stop()}catch{}}setLiveInterim('');setLiveState('paused')};
 const resumeLive=()=>{setLiveError('');const recognition=createRecognition();if(!recognition)return;try{recognition.start();setLiveState('listening')}catch{setLiveError('No se pudo reanudar la escucha.')}};
 const stopLive=()=>{const recognition=recognitionRef.current;recognitionRef.current=null;if(recognition){try{recognition.stop()}catch{}}setLiveInterim('');setLiveState('finished')};
 const resetLive=()=>{const recognition=recognitionRef.current;recognitionRef.current=null;if(recognition){try{recognition.abort()}catch{}}setLiveFinal('');setLiveInterim('');setLiveError('');setLiveState('idle')};
 const liveText=(liveFinal+(liveInterim?`${liveFinal?' ':''}${liveInterim}`:'')).trim();
 const copyLive=async()=>{if(!liveText)return;try{await navigator.clipboard.writeText(liveText)}catch{}};

 return <div className="app"><header><div><div className="brand"><Mic2 size={22}/> ActaAI</div><p>Convierte conversaciones en información útil.</p></div><span className="badge">Análisis de reuniones</span></header>
 <main>{!data?<section className="hero"><div className="heroText"><span className="eyebrow">TRANSCRIPCIÓN · PARTICIPANTES · RESUMEN</span><h1>{mode==='live'?'Escucha en vivo.\nTranscribe al instante.':'Sube un audio.\nEntiende toda la conversación.'.split('\n').map((x,i)=><React.Fragment key={x}>{x}{i===0&&<br/>}</React.Fragment>)}</h1>{mode==='live'?<p>Activa el micrófono y ve apareciendo la transcripción mientras hablas. Puedes pausar, reanudar, copiar o descargar el texto.</p>:<p>Obtén una transcripción ordenada, identifica participantes, revisa temas, acuerdos y próximos pasos en un solo lugar.</p>}</div><div className="uploadCard"><div className="modeSwitch"><button className={mode==='upload'?'selected':''} onClick={()=>setMode('upload')}><Upload size={16}/> Subir audio</button><button className={mode==='live'?'selected':''} onClick={()=>setMode('live')}><Radio size={16}/> En vivo</button></div>
 {mode==='upload'?<><label className="drop"><Upload size={35}/><strong>{file?file.name:'Selecciona un audio'}</strong><span>MP3, WAV, M4A, MP4, OGG o FLAC</span><input type="file" accept="audio/*,video/mp4" onChange={e=>{setFile(e.target.files?.[0]||null);setError('')}}/></label>{file&&<div className="fileRow"><FileAudio/><div><b>{file.name}</b><small>{(file.size/1024/1024).toFixed(1)} MB</small></div></div>}<button className="primary" disabled={!file||busy} onClick={analyze}>{busy?'Analizando audio…':'Analizar audio'}</button>{busy&&<small className="hint">El análisis puede tardar varios minutos según la duración del archivo.</small>}{error&&<div className="errorBox"><AlertCircle size={18}/><span>{error}</span></div>}</>:
 <div className="liveCard"><div className={`liveStatus ${liveState}`}><span className="pulseDot"/><div><strong>{liveState==='listening'?'Escuchando en vivo':liveState==='paused'?'Escucha pausada':liveState==='finished'?'Transcripción finalizada':'Micrófono listo'}</strong><small>{liveState==='listening'?'Habla con normalidad. El texto aparecerá debajo.':'El audio se procesa localmente mediante el reconocimiento del navegador.'}</small></div></div><div className="liveTranscript" contentEditable={liveState==='finished'} suppressContentEditableWarning>{liveText||<span className="placeholder">La transcripción aparecerá aquí en tiempo real…</span>}</div><div className="liveControls">{liveState==='idle'&&<button className="primary livePrimary" onClick={startLive}><Mic2 size={18}/> Iniciar escucha</button>}{liveState==='listening'&&<><button className="secondaryBtn" onClick={pauseLive}><Pause size={17}/> Pausar</button><button className="dangerBtn" onClick={stopLive}><Square size={17}/> Finalizar</button></>}{liveState==='paused'&&<><button className="primary compact" onClick={resumeLive}><Mic2 size={17}/> Reanudar</button><button className="dangerBtn" onClick={stopLive}><Square size={17}/> Finalizar</button></>}{liveState==='finished'&&<><button className="secondaryBtn" onClick={copyLive}><Copy size={17}/> Copiar</button><button className="secondaryBtn" onClick={()=>downloadText(liveText,'transcripcion-en-vivo.txt')} disabled={!liveText}><Download size={17}/> Descargar</button><button className="secondaryBtn" onClick={resetLive}><RotateCcw size={17}/> Nueva sesión</button></>}</div>{liveError&&<div className="errorBox"><AlertCircle size={18}/><span>{liveError}</span></div>}<small className="hint liveHint">Funciona mejor en Chrome o Edge. El navegador pedirá permiso para usar el micrófono.</small></div>}</div></section>:
 <section><div className="topline"><div><button className="back" onClick={()=>{setData(null);setQuery('')}}>← Nuevo audio</button><h2>{data.title}</h2><p>{data.duration} · {data.speakers.length} participante{data.speakers.length===1?'':'s'} detectado{data.speakers.length===1?'':'s'}</p></div><div className="actions"><button className="export" onClick={exportTxt}><Download size={16}/> Exportar acta</button><button className="export" onClick={()=>sendEmail('transcript')}><Mail size={16}/> Enviar transcripción</button><button className="export" onClick={()=>sendEmail('full')}><Mail size={16}/> Enviar acta completa</button></div></div>
 <nav>{[['resumen','Resumen'],['transcripcion','Transcripción'],['participantes','Participantes'],['tareas','Acuerdos y tareas']].map(([k,l])=><button className={tab===k?'active':''} onClick={()=>setTab(k)} key={k}>{l}</button>)}</nav>
 {tab==='resumen'&&<div className="grid"><div className="card wide"><h3>Resumen ejecutivo</h3><p className="summary">{data.summary}</p></div><div className="card"><h3>Temas tratados</h3>{data.topics.length?data.topics.map((t,i)=><div className="topic" key={t}><span>{i+1}</span>{t}</div>):<p className="muted">No se detectaron temas suficientes.</p>}</div><div className="card"><h3>Participación</h3>{data.speakers.map(s=><div className="particip" key={s.id}><div><b>{s.name}</b><small>{s.time} · {s.turns} intervenciones</small></div><strong>{s.pct}%</strong><div className="bar"><i style={{width:s.pct+'%'}}/></div></div>)}</div><div className="card wide"><h3>Decisiones y compromisos</h3>{data.agreements.length?data.agreements.map((a,i)=><div className="agreement" key={i}><CheckCircle2/><div><b>{a.text}</b><small>Responsable: {a.owner} · {a.due}</small></div></div>):<p className="muted">No se detectaron compromisos explícitos en la conversación.</p>}</div></div>}
 {tab==='transcripcion'&&<div className="card"><div className="search"><Search/><input placeholder="Buscar en la conversación…" value={query} onChange={e=>setQuery(e.target.value)}/></div><div className="transcript">{visible.map((x,i)=><div className="line" key={`${x.start}-${i}`}><button className="time"><PlayCircle size={14}/>{x.time}</button><div><b>{x.speaker}</b><p>{x.text}</p></div></div>)}</div></div>}
 {tab==='participantes'&&<div className="people">{data.speakers.map(s=><div className="person card" key={s.id}><div className="avatar">{s.name[0]}</div><h3>{s.name}</h3><span>Participante detectado</span><div className="stats"><div><strong>{s.time}</strong><small>hablando</small></div><div><strong>{s.pct}%</strong><small>participación</small></div><div><strong>{s.turns}</strong><small>intervenciones</small></div></div><button onClick={()=>renameSpeaker(s.name)}><Edit3 size={14}/> Renombrar participante</button></div>)}</div>}
 {tab==='tareas'&&<div className="card"><h3>Acuerdos y tareas detectadas</h3>{data.agreements.length?data.agreements.map((a,i)=><div className="task" key={i}><span className="num">{i+1}</span><div><b>{a.text}</b><p><Users size={14}/> {a.owner} <Clock3 size={14}/> {a.due}</p></div><select defaultValue="pendiente"><option value="pendiente">Pendiente</option><option>En curso</option><option>Completada</option></select></div>):<p className="muted">No se detectaron tareas explícitas.</p>}</div>}
 </section>}</main></div>}
createRoot(document.getElementById('root')).render(<App/>);
