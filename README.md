# Happy_Chickens
Python Skripte zum Video-tracking von Hühnern.

Die Skripte brauchen eine aktuelle opencv-python Version, mehr nicht.

Dazu muss OpenCV für Python installiert werden. 
Entweder:
pip install opencv-python
Oder:
pip install opencv-contrib-python

Letzteres ist nötig um wenigstens einige Tracker in OpenCV nutzen zu können.

Es fehlen leider hier die Gewichte für das yolo-coco DeepNet, weil diese Dateien viel zu groß für github sind. Diese Dateien müssen in die Ordner für yolo-coco-v3 und yolo-coco-v4 eingefügt werden. Der Name der Datei für die Gewichte muss yolo-coco.weights lauten.

Die Skripte können auf der Commandline-Ebene wie folgt aufgerufen werden:

Skript nur für Bilder:
python yolo_image.py --image [Pfad zum Bild] --yolo [Pfad zur YOLO Konfiguration, den Gewichten und den Klassennamen des coco Datensatzes]

Skript für Videos:
python yolo_video.py --input [Pfad zum Video] --output [Pfad für den Output] --yolo [Pfad zur YOLO Konfiguration, den Gewichten und den Klassennamen des coco Datensatzes]

In den Ordner Images und Videos sind Beispiele zur Objektdetektion von Hühnern.

Skript zum Tracking:
python simple_opencv_tracking.py --video [Pfad zum Video] --tracker [Trackertyp, Standard: kfc]

Das zu trackende Object muss dann manual mit einem Rechteck markiert werden. Das Skript kann mit ESC gestoppt werden.

