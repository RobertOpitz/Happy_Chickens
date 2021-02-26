# Happy_Chickens
Python Skripte zum Video-tracking von Hühnern.

Es fehlen leider hier die Gewichte für das yolo-coco DeepNet, weil diese Dateien viel zu groß für github sind. Diese Dateien müssen in die Ordner für yolo-coco-v3 und yolo-coco-v4 eingefügt werden.

Die Skripte brauchen eine aktuelle opencv-python Version, mehr nicht.

Die Skripte können auf der Commandline-Ebene wie folgt aufgerufen werden:

Skript nur für Bilder:
python yolo_image.py --image [Pfad zum Bild] --yolo [Pfad zur YOLO Konfiguration, den Gewichten und den Klassennamen des coco Datensatzes]

Skript für Videos:
python yolo_video.py --input [Pfad zum Video] --output [Pfad für den Output] --yolo [Pfad zur YOLO Konfiguration, den Gewichten und den Klassennamen des coco Datensatzes]

In den Ordner Images und Videos sind Beispiele zur Objektdetektion von Hühnern.

