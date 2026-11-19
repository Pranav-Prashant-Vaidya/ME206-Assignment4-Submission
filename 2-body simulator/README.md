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

---

## 📋 Software Requirements & Dependencies

### Prerequisites / System Requirements
* **Operating System:** Any modern operating system (Windows, macOS, Linux, ChromeOS, iOS, Android).
* **Web Browser:** Any modern web browser with HTML5 Canvas and JavaScript enabled (e.g., Google Chrome, Mozilla Firefox, Microsoft Edge, Apple Safari).
* **Hardware:** No dedicated GPU required; standard integrated graphics are sufficient.

### Dependencies & External Libraries
* **Zero Local Build Dependencies:** No Node.js, Python, or complex build tools (`npm`, `webpack`) are required to install or configure.
* **Tailwind CSS (v3.x via CDN):** Included directly via CDN (`https://cdn.tailwindcss.com`) for modern UI styling.
* **Google Fonts API:** Included via HTML link tag for the *Inter* typography font.

---

## 📥 Installation Steps

You can set up this simulator on your local machine using either of the following methods:

### Method A: Download directly (No Git required)
1. Go to the main page of this GitHub repository.
2. Click the green **Code** button at the top right.
3. Select **Download ZIP**.
4. Extract/unzip the downloaded file to any folder on your computer.

### Method B: Clone via Git Command Line
Open your terminal or command prompt and run:
```bash
git clone https://github.com/your-username/orbit-dynamics-simulator.git
cd orbit-dynamics-simulator
```

---

## 🚀 How to Run the Simulation

### Option 1: Run Locally in Browser (Fastest)
1. Open the project folder on your computer.
2. Double-click the `index.html` file (or right-click and choose **Open with** > your preferred browser).
3. The simulator will launch immediately in full-screen mode!

### Option 2: Host & Run via GitHub Pages
1. Push this repository to your public GitHub account.
2. Go to your repository's **Settings** tab $\rightarrow$ select **Pages** from the left sidebar.
3. Under **Build and deployment** $\rightarrow$ **Branch**, select `main` (or `master`) and click **Save**.
4. GitHub will generate a live web link (e.g., `https://your-username.github.io/orbit-dynamics-simulator/`) where anyone can access and run the simulation directly online.

---

## 🎮 Controls & Keyboard Shortcuts

| Action | Control / Shortcut |
| :--- | :--- |
| **Pan Camera** | Click + Drag anywhere on space |
| **Zoom In / Out** | Mouse Scroll Wheel / Touchpad Pinch |
| **Fast-Forward Speed** | `UP` Arrow Key / Time Warp Slider |
| **Slow Down Speed** | `DOWN` Arrow Key / Time Warp Slider |
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
