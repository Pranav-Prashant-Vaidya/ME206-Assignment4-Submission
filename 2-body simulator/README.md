# 🪐 OrbitDynamics 2-Body Gravitational Simulator

An interactive, high-performance web-based 2-body orbital mechanics simulator built using HTML5 Canvas, JavaScript, and Tailwind CSS. It allows users to simulate classical Newtonian gravitational interactions between two celestial bodies in real time, view predicted full orbital trajectory loops, adjust time warp speed, and change reference frames.

---

## ✨ Features

* **Full Orbital Trajectory Prediction:** Computes and renders complete closed elliptical or open hyperbolic predicted paths ahead of time rather than just leaving past trails.
* **Fast-Forward & Time Warp:** Adjust simulation speed dynamically using onscreen controls or keyboard shortcuts (`UP` and `DOWN` arrow keys).
* **Multiple Reference Frames:** Switch seamlessly between:
  * **Center of Mass (Barycenter):** Inertial frame around the shared center of mass.
  * **Body 1 Stationary:** See the relative orbit of Body 2 around Body 1.
  * **Body 2 Stationary:** See the relative orbit of Body 1 around Body 2.
* **Interactive Controls & Navigation:**
  * Drag to pan around space.
  * Scroll wheel to zoom in/out.
  * Recenter button to reset camera view.
* **Real-time Telemetry HUD:** Live readouts for body speeds, orbital separation ($r$), and estimated orbital period ($T$).
* **Zero Dependencies / Standalone:** Pure front-end application with no complex build steps required.

---

## 🚀 Quick Start & How to Run

### Option 1: Open Directly in Browser
1. Clone or download this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/orbit-dynamics-simulator.git
   ```
2. Navigate into the directory and double-click `index.html` to open it in any modern browser (Chrome, Edge, Firefox, Safari).

### Option 2: Host via GitHub Pages (Recommended)
1. Push this code to a public GitHub repository.
2. Go to **Settings** > **Pages** in your repository menu.
3. Under **Branch**, select `main` (or `master`) and click **Save**.
4. GitHub will give you a live link to share your interactive simulator on the web!

---

## 🎮 Controls & Shortcuts

| Action | Control |
| :--- | :--- |
| **Pan Camera** | Click + Drag anywhere on space |
| **Zoom In / Out** | Mouse Scroll Wheel |
| **Fast-Forward Speed** | `UP` Arrow Key / Slider |
| **Slow Down Speed** | `DOWN` Arrow Key / Slider |
| **Pause / Play** | `SPACE` Key / Toggle Button |
| **Recenter Camera** | Click "Recenter" Button |

---

## 🛠️ Built With

* **HTML5 Canvas API** — Physics rendering engine and smooth trajectory calculations.
* **JavaScript (ES6+)** — Real-time Euler sub-step numerical integration engine.
* **Tailwind CSS** — Modern dark-mode UI overlay styling.

---

## 📝 License

This project is open-source under the [MIT License](LICENSE). Feel free to fork, customize, and share!
