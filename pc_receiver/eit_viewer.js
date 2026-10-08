/* Optional EIT panel for the shared RGB/pose/IMU playback timeline. */
(async function () {
  const root = document.currentScript.dataset.eitRoot;
  const el = id => document.getElementById(id);
  const vid = el('video');
  try {
    const response = await fetch(root + '/metadata.json');
    if (!response.ok) throw new Error('EIT metadata unavailable');
    const m = await response.json();
    const binary = await fetch(root + '/' + m.binaryFile);
    if (!binary.ok) throw new Error('EIT measurement data unavailable');
    const values = new Float32Array(await binary.arrayBuffer());
    if (values.length !== m.frames * 256) throw new Error('EIT data size mismatch');
    const times = m.rawRelativeSeconds, means = new Float32Array(m.frames);
    const baseline = m.baselineMagnitude.flat();
    for (let i = 0; i < m.frames; i++) {
      let sum = 0, n = 0;
      for (let j = 0; j < 256; j++) {
        const x = values[i * 256 + j];
        if (Number.isFinite(x)) { sum += x; n++; }
      }
      means[i] = n ? sum / n : NaN;
    }
    const meanMaximum=Math.max(1e-8,...means.filter(Number.isFinite));
    el('eitOffset').value = m.offsetMs;
    const clockMessage = m.clockStatus === 'phone_aligned'
      ? 'EIT timestamps are declared to be on the iPhone timeline.'
      : 'EIT clock alignment is unverified. Display uses recorded Unix timestamps plus the offset below.';
    el('eitClock').textContent = clockMessage + ' Matching timestamps does not verify sensor latency.';
    function nearestIndex(t) {
      let lo = 0, hi = times.length;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (times[mid] < t) lo = mid + 1; else hi = mid; }
      if (!lo) return 0;
      if (lo === times.length) return lo - 1;
      return Math.abs(times[lo] - t) < Math.abs(times[lo - 1] - t) ? lo : lo - 1;
    }
    function context(id) {
      const canvas = el(id), ratio = devicePixelRatio || 1;
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (canvas.width !== Math.round(w * ratio) || canvas.height !== Math.round(h * ratio)) {
        canvas.width = Math.round(w * ratio); canvas.height = Math.round(h * ratio);
      }
      const g = canvas.getContext('2d'); g.setTransform(ratio, 0, 0, ratio, 0, 0);
      g.clearRect(0, 0, w, h); g.font = '12px system-ui'; g.fillStyle = '#b9c7db';
      return [g, w, h];
    }
    function color(x, relative) {
      if (!Number.isFinite(x)) return '#59616b';
      if (relative) {
        const t = Math.min(1, Math.abs(x) / 20);
        return x < 0 ? `rgb(${Math.round(235-185*t)},${Math.round(235-100*t)},235)`
          : `rgb(235,${Math.round(235-135*t)},${Math.round(235-175*t)})`;
      }
      const t = Math.max(0, Math.min(1, (x-m.magnitudeRange[0])/(m.magnitudeRange[1]-m.magnitudeRange[0])));
      return `rgb(${Math.round(25+225*t)},${Math.round(45+175*Math.sin(t*Math.PI/2))},${Math.round(115-65*t)})`;
    }
    function draw() {
      const t = vid.currentTime || 0, offset = Number(el('eitOffset').value) / 1000;
      if (!Number.isFinite(offset)) return;
      const i = nearestIndex(t-offset), error = times[i]+offset-t;
      const matched = Math.abs(error) <= m.matchToleranceSeconds;
      const relative = el('eitMode').value === 'relative';
      const [g,w,h] = context('eitFrame');
      const size = Math.min(w-90,h-65), cell = size/16, left = (w-size)/2+15, top = 20;
      g.textAlign = 'center';
      if (!matched) {
        g.fillText('No EIT frame at this time',w/2,h/2);
      } else {
        for (let r=0;r<16;r++) for (let c=0;c<16;c++) {
          const j=r*16+c, raw=values[i*256+j];
          const value=relative ? (baseline[j] > 1e-8 ? 100*(raw/baseline[j]-1) : NaN) : raw;
          g.fillStyle=color(value,relative);g.fillRect(left+c*cell,top+r*cell,cell-.5,cell-.5);
        }
        g.fillStyle='#b9c7db';
        for(let c=0;c<16;c+=2)g.fillText(String(c),left+(c+.5)*cell,top+size+17);
        g.textAlign='right';
        for(let r=0;r<16;r+=2)g.fillText(m.injectionPairs[r].join('-'),left-7,top+(r+.65)*cell);
        g.textAlign='center';g.fillText('Measurement index',left+size/2,top+size+35);
        g.save();g.translate(left-47,top+size/2);g.rotate(-Math.PI/2);g.fillText('Injection pair',0,0);g.restore();
      }
      el('eitInfo').textContent=matched
        ? `Frame ${m.frameIndices[i]} | ${(m.frequencyHz/1000).toFixed(1)} kHz | offset from cursor ${(error*1000).toFixed(1)} ms | missing cells: ${m.missingCells[i]}`
        : `No frame within ${(1000*m.matchToleranceSeconds).toFixed(1)} ms. The last frame is not reused.`;
      el('eitFrame').dataset.frameIndex=matched ? m.frameIndices[i] : '';
      el('eitFrame').dataset.matched=String(matched);
      const [p,pw,ph]=context('eitTrace');
      const maximum=meanMaximum;
      const duration=m.recordingDurationSeconds;
      const X=s=>45+(s+offset)/duration*(pw-60),Y=v=>ph-30-v/maximum*(ph-55);
      p.save();p.beginPath();p.rect(45,15,pw-60,ph-45);p.clip();
      p.strokeStyle='#65d5e6';p.lineWidth=1.2;p.beginPath();let connected=false;
      for(let k=0;k<times.length;k++) {
        if(!Number.isFinite(means[k])) {connected=false;continue;}
        if(!connected || (k && times[k]-times[k-1]>.1))p.moveTo(X(times[k]),Y(means[k]));
        else p.lineTo(X(times[k]),Y(means[k]));connected=true;
      }
      p.stroke();p.strokeStyle='#fff';p.beginPath();p.moveTo(X(t-offset),15);p.lineTo(X(t-offset),ph-30);p.stroke();p.restore();
      p.fillStyle='#b9c7db';p.fillText('Mean magnitude',8,12);p.fillText(maximum.toFixed(2),3,30);p.fillText('0',25,ph-28);
      for(let k=0;k<=4;k++){const time=k*duration/4;p.fillText(time.toFixed(0)+' s',45+time/duration*(pw-60)-10,ph-8);}
      g.fillStyle='#b9c7db';g.textAlign='left';
      g.fillText(relative?'Color range: -20% to +20%':'Color range: '+m.magnitudeRange.map(x=>x.toPrecision(3)).join(' to '),8,h-4);
    }
    el('eitMode').onchange=draw;el('eitOffset').oninput=draw;
    vid.addEventListener('timeupdate',draw);vid.addEventListener('seeked',draw);window.addEventListener('resize',draw);
    let last=-1;
    function tick(){if(!vid.paused && vid.currentTime!==last){last=vid.currentTime;draw();}requestAnimationFrame(tick);}
    requestAnimationFrame(tick);draw();
    el('eitSection').dataset.loaded='true';
  } catch(error) {el('eitInfo').textContent='EIT display error: '+error.message;console.error(error);}
})();
