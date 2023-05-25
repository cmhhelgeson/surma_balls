import './style.css'
import { setupCounter } from './counter.ts'

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div>
    <h1>WebGPU Ball Demonstration</h1>
    <canvas id="canvas" width="1000" height="800"></canvas>
    <a href="https://surma.dev/things/webgpu/">Tutorial</a>
  </div>
`

setupCounter(document.querySelector<HTMLButtonElement>('#counter')!)
