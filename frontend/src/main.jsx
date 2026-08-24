import React,{useMemo,useState} from 'react';
import {createRoot} from 'react-dom/client';
import {Upload,FileAudio,Users,CheckCircle2,Clock3,Mic2,Search,Download,PlayCircle} from 'lucide-react';
import './styles.css';

const demo={
 title:'Reunión de proyecto — Portal de Reservas',duration:'38:42',date:'24/08/2026',
 summary:'El equipo revisó el estado del portal de reservas, los problemas de usabilidad detectados en formularios y el flujo de cancelación por fuerza mayor. Se acordó corregir la pérdida de foco en campos de texto, completar la información visible para Oficina Central y cerrar las mejoras de administración antes de preparar la siguiente versión.',
 topics:['Formularios y experiencia de usuario','Cancelación por fuerza mayor','Administración de inmuebles','Tarifas y permisos','Próxima versión'],
 agreements:[
  {text:'Corregir la pérdida de foco en todos los campos de texto.',owner:'Alfredo',due:'Próxima versión'},
  {text:'Mostrar fundamento y archivo adjunto en el detalle de Oficina Central.',owner:'Equipo desarrollo',due:'Próxima versión'},
  {text:'Incorporar creación de inmuebles y modificación de tarifas.',owner:'Equipo desarrollo',due:'Pendiente validación'}
 ],
 speakers:[
  {id:'SPEAKER_00',name:'Alfredo',time:'18:12',pct:47,turns:31,topics:'prioridades, UX, reglas de negocio'},
  {id:'SPEAKER_01',name:'Carolina',time:'12:05',pct:31,turns:22,topics:'administración, cancelaciones'},
  {id:'SPEAKER_02',name:'Pedro',time:'08:25',pct:22,turns:14,topics:'desarrollo, implementación'}
 ],
 transcript:[
  {time:'00:00:12',speaker:'Alfredo',text:'Revisemos primero los problemas que encontramos en los formularios de confirmación.'},
  {time:'00:00:22',speaker:'Carolina',text:'Cuando escribo en Fundamento de la confirmación, después de cada letra el campo pierde el foco.'},
  {time:'00:00:38',speaker:'Pedro',text:'Eso parece venir de un re-render del componente. Lo podemos corregir para todos los campos de texto.'},
  {time:'00:01:02',speaker:'Alfredo',text:'Perfecto. También quiero que en fuerza mayor se vea en Oficina Central el texto ingresado y el archivo solicitado.'},
  {time:'00:01:18',speaker:'Carolina',text:'Sí, porque hoy se pide la información al usuario, pero después no queda disponible en el detalle.'},
  {time:'00:01:45',speaker:'Alfredo',text:'Después revisemos administración. Falta poder crear inmuebles y modificar tarifas.'},
  {time:'00:02:01',speaker:'Pedro',text:'Lo agrego dentro de la siguiente versión junto con las mejoras pendientes.'}
 ]
};

function App(){
 const [file,setFile]=useState(null),[tab,setTab]=useState('resumen'),[query,setQuery]=useState(''),[data,setData]=useState(null),[busy,setBusy]=useState(false);
 const analyze=()=>{if(!file)return;setBusy(true);setTimeout(()=>{setData({...demo,title:file.name});setBusy(false)},1200)};
 const visible=useMemo(()=>!data?[]:data.transcript.filter(x=>(x.text+' '+x.speaker).toLowerCase().includes(query.toLowerCase())),[data,query]);
 return <div className="app"><header><div><div className="brand"><Mic2 size={22}/> ActaAI</div><p>Convierte conversaciones en información útil.</p></div><span className="badge">MVP · v0.1</span></header>
 <main>{!data?<section className="hero"><div className="heroText"><span className="eyebrow">TRANSCRIPCIÓN + DIARIZACIÓN + RESUMEN</span><h1>Sube un audio.<br/>Entiende toda la reunión.</h1><p>Identifica quién habló, qué dijo, qué se decidió y cuáles son los próximos pasos.</p></div><div className="uploadCard"><label className="drop"><Upload size={35}/><strong>{file?file.name:'Selecciona un audio'}</strong><span>MP3, WAV, M4A o MP4</span><input type="file" accept="audio/*,video/mp4" onChange={e=>setFile(e.target.files?.[0]||null)}/></label>{file&&<div className="fileRow"><FileAudio/><div><b>{file.name}</b><small>{(file.size/1024/1024).toFixed(1)} MB</small></div></div>}<button className="primary" disabled={!file||busy} onClick={analyze}>{busy?'Analizando audio…':'Analizar audio'}</button><small className="hint">Esta versión incluye modo demostración. El backend queda preparado para Whisper + diarización real.</small></div></section>:
 <section><div className="topline"><div><button className="back" onClick={()=>setData(null)}>← Nuevo audio</button><h2>{data.title}</h2><p>{data.date} · {data.duration} · {data.speakers.length} participantes</p></div><button className="export"><Download size={16}/> Exportar acta</button></div>
 <nav>{[['resumen','Resumen'],['transcripcion','Transcripción'],['participantes','Participantes'],['tareas','Acuerdos y tareas']].map(([k,l])=><button className={tab===k?'active':''} onClick={()=>setTab(k)} key={k}>{l}</button>)}</nav>
 {tab==='resumen'&&<div className="grid"><div className="card wide"><h3>Resumen ejecutivo</h3><p className="summary">{data.summary}</p></div><div className="card"><h3>Temas tratados</h3>{data.topics.map((t,i)=><div className="topic" key={t}><span>{i+1}</span>{t}</div>)}</div><div className="card"><h3>Participación</h3>{data.speakers.map(s=><div className="particip" key={s.id}><div><b>{s.name}</b><small>{s.time} · {s.turns} intervenciones</small></div><strong>{s.pct}%</strong><div className="bar"><i style={{width:s.pct+'%'}}/></div></div>)}</div><div className="card wide"><h3>Decisiones y compromisos</h3>{data.agreements.map(a=><div className="agreement" key={a.text}><CheckCircle2/><div><b>{a.text}</b><small>Responsable: {a.owner} · {a.due}</small></div></div>)}</div></div>}
 {tab==='transcripcion'&&<div className="card"><div className="search"><Search/><input placeholder="Buscar en la conversación…" value={query} onChange={e=>setQuery(e.target.value)}/></div><div className="transcript">{visible.map((x,i)=><div className="line" key={i}><button className="time"><PlayCircle size={14}/>{x.time}</button><div><b>{x.speaker}</b><p>{x.text}</p></div></div>)}</div></div>}
 {tab==='participantes'&&<div className="people">{data.speakers.map(s=><div className="person card" key={s.id}><div className="avatar">{s.name[0]}</div><h3>{s.name}</h3><span>{s.id}</span><div className="stats"><div><strong>{s.time}</strong><small>hablando</small></div><div><strong>{s.pct}%</strong><small>participación</small></div><div><strong>{s.turns}</strong><small>intervenciones</small></div></div><p><b>Temas:</b> {s.topics}</p><button>Renombrar hablante</button></div>)}</div>}
 {tab==='tareas'&&<div className="card"><h3>Acuerdos y tareas detectadas</h3>{data.agreements.map((a,i)=><div className="task" key={i}><span className="num">{i+1}</span><div><b>{a.text}</b><p><Users size={14}/> {a.owner} <Clock3 size={14}/> {a.due}</p></div><select defaultValue="pendiente"><option value="pendiente">Pendiente</option><option>En curso</option><option>Completada</option></select></div>)}</div>}
 </section>}</main></div>
}
createRoot(document.getElementById('root')).render(<App/>);
