# Subgruppensuche durch lokales Modelltraining

Dieses Repo enthält die **modifizierte `pysubgroup`-Bibliothek** aus der Arbeit und
schrittweise die Skripte und Daten, mit denen die berichteten Experimente
nachvollzogen werden können.

## Warum eine lokale `pysubgroup`?

Die Arbeit erweitert [pysubgroup](https://github.com/flemmerich/pysubgroup)
(Apache 2.0). Die zusätzlichen Module liegen nur in `./pysubgroup`, nicht
auf PyPI:

- `pysubgroup.model_adaptability_target`
- `pysubgroup.parallel_model_adaptability_dfs`

`requirements.txt` installiert `pysubgroup` daher **nicht** von PyPI.
Anschließend wird die lokale Kopie **editierbar** (`-e`) installiert, sodass
`import pysubgroup` auf `pysubgroup/src/pysubgroup` zeigt.

## Einrichtung (Python 3.10+)

Unter Debian/Ubuntu zuerst `python3-venv` installieren
(`sudo apt install python3.10-venv` bzw. das Paket zur verwendeten
Python-Version), danach die virtuelle Umgebung anlegen.

```bash
git clone https://github.com/Jadotville/submod-bachelor-thesis.git
cd submod-bachelor-thesis

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -U pip "setuptools>=46.1.0,<81" "setuptools_scm[toml]>=5" wheel
pip install -r requirements.txt

# Lokale Bibliothek (nicht PyPI). Die Versionsvariable ist nötig, weil
# setuptools-scm in diesem Unterordner keine Version erkennen kann.
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYSUBGROUP=0.9.0 pip install -e ./pysubgroup
```

`setuptools<81` ist erforderlich: Die Upstream-`pysubgroup` importiert noch
`pkg_resources`, das neuere setuptools-Versionen entfernt haben.

### Prüfen, dass die lokale Kopie aktiv ist

```bash
python -c "import pysubgroup; print(pysubgroup.__file__)"
```

Der Pfad muss `pysubgroup/src/pysubgroup` enthalten. Zeigt er ins
`.venv/.../site-packages` ohne diesen Ordner, wurde versehentlich die
PyPI-Version installiert. Dann `pip uninstall pysubgroup` und die
editierbare Installation oben erneut ausführen.

Optionale Bibliothekstests:

```bash
python -m pytest pysubgroup/tests -q
```
