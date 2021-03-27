# Entropia.py

Este paquete centraliza diferentes tipos de entropias de la información.

# Descarga e instalación:

Para utilizarlo se debe contar con `Cython`.
La forma de instalar `Entropia.py` y sus dependencias se muestra a continuación:

```
git clone https://github.com/jbarberia/Entropia.py
cd Entropia.py
pip install -r requirements.txt
python setup.py install
```

Aunque es preferible instalarlo por medio del sistema de gestión de paquetes pip:
```
pip install cython
pip install git+https://github.com/jbarberia/Entropia.py
```

# Uso:

Su uso es bastante intuitivo:
``` python
import entropia

x = [4, 7, 9, 10, 6, 11, 3]
e1 = entropia.bandt_and_pompe(x, m=3, t=1)
e2 = entropia.bandt_and_pompe_normal(x, m=3, t=1)
e3 = entropia.weight_entropy(x, m=3, t=1)
```
Siendo `m` la dimensión de embedding y `t` el delay de muestreo de la señal.
