# Self-Righting Boat

This project explores the optimized design of self-righting boats, including the development of mathematical models and software tools to support this endeavor.

---

## 📜 Context

Most ships lack the ability to self-right, making them vulnerable to capsizing under the influence of large waves and propulsion forces. Smaller and faster ships, in particular, are at higher risk.

While rescue boats and some sailing ships (equipped with heavy keels) are designed to self-right, other types of vessels, such as kiteboats, could benefit significantly from this capability. Kiteboats, subjected to strong pulling forces from kites, lack long, heavy keels and thus present unique design challenges.

This project aims to address these challenges and contribute to the safety and reliability of kiteboats and similar vessels.

---

## 🛠️ Installation
You can download the code directly from GitHub or using git:

```bash
git clone git@github.com:Azmadax/self-righting-boat.git
```

To install the required dependencies, follow the steps below:

1. Install `uv` by following the official [installation guide](https://docs.astral.sh/uv/getting-started/installation).
2. Alternatively, run the command from root of repository:  
   ```bash
   cd self-righting-boat
   make install-uv
   ```

---

## 🚀 Usage

To launch an example, run the following command from root of repository:  
```bash
uv sync --package hydrostatic
uv run examples/hydrostatic_2d_example.py
```
Alternatively, if you have make install, just run:
```bash
make demo
```
![circle example](docs/source/img/circle_example.png)
![catamaran example](docs/source/img/catamaran.png)
![catamaran GZ curve example](docs/source/img/GZ_curve.png)
## Test
To launch the complete suite of tests, launch the following command:
```bash
uv sync --package hydrostatic
uv run pytest packages/hydrostatic/tests
```
Alternatively, if you have make install, just run:
   ```bash
   make test
   ```


---

## 📝 Features

- **Mathematical modeling:** Tools for analyzing hydrostatic and dynamic stability.
- **Software support:** Examples and scripts for evaluating boat designs.
- **Applications:** Tailored solutions for kiteboats and other vessels requiring self-righting capabilities.

---

## 📜 Documentation
Documentation is available at:
https://self-righting-boat.readthedocs.io/en/latest/

## 🛠️ Development Roadmap

The project was started in November 2024 and is still at an early stage.  
The optimization is expected to be ready by March 2025, at the end of the student project.  
The developments might be continued afterward.

Features:
- [x] Center of buoyancy computation
- [x] Vertical equilibrium
- [x] Placement of center of gravity
- [x] Righting arm (GZ) computation
- [x] GZ curve computation
- [x] Research of equilibrium points
- [ ] Metacentric height computation at equilibrium points using waterplane
- [ ] Optimized additional low density static superstructure to enable self-righting
- [ ] Optimized high density static keel to enable self-righting 

Supported geometries:
- [x] single 2D closed polygon
- [x] multiple 2D closed polygons
- [ ] single 2D NURBS (numerical integration)
- [ ] 3D meshes
- [ ] 2D polygons with closed holes partially filled (free surface effect)

## 📚 Credits

This project was initiated as part of the Ocean Options program at **École Centrale Nantes**.
It was proposed by the **International Kiteboat Federation France (IKIFF)** [kiteboat.org], with the goal of improving the safety of kiteboats.

### Project Team:
Students:
- **Nicolas Fihey**  
- **Lucas Vieillefond**  
- **Thimothée Leroy**  

Advisers: 
- **Baptiste Labat**
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue?logo=linkedin&logoWidth=20&style=flat-square)](https://www.linkedin.com/in/baptiste-labat-01751138/)
[![GitHub](https://img.shields.io/badge/-GitHub-black?logo=github&logoWidth=20&style=flat-square)](https://github.com/baptistelabat)

---

## 🤝 Contributions

Contributions are welcome! Please open an issue or submit a pull request to suggest changes, report bugs, or propose new features.

Here are a few code guidelines:  
- We use english for code and comments.
- We use google style docstring.
- We use type hinting.  
Please be sure to install the pre-commit tool in order to check your code while commiting in order to keep a clean project history.
```bash
pre-commit install
```
Please have a look to makefile to find helpful commands.

## 📜 License
![License](https://img.shields.io/badge/license-MPL%202.0-brightgreen)

This project is licensed under the Mozilla Public License 2.0 - see the [LICENSE](./LICENSE) file for details.

---

## 🖇️ References

- [International Kiteboat Federation France (IKIFF)](https://kiteboat.org)  
- [Option Ocean at Centrale Nantes](https://www.ec-nantes.fr/formation/les-options-de-2e-et-3e-annee/option-ocean)
