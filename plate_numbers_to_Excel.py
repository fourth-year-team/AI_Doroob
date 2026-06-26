import os

# 1. حدد مسار المجلد الذي يحتوي على الـ 60 صورة بعد إعادة تسميتها
dataset_folder = "./vv"  # ضع اسم مجلدك هنا
output_file = "train_label.txt"

# الامتدادات المدعومة للصور
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

if not os.path.exists(dataset_folder):
    print(f"❌ خطأ: المجلد '{dataset_folder}' غير موجود. يرجى التأكد من المسار.")
else:
    # فتح ملف النصي بصيغة UTF-8 لدعم المحارف والشحطات بشكل سليم
    with open(output_file, "w", encoding="utf-8") as f:
        counter = 0
        
        for filename in os.listdir(dataset_folder):
            # فحص اللاحقة للتأكد من أنها صورة
            name_part, ext_part = os.path.splitext(filename)
            if ext_part.lower() in valid_extensions:
                
                # اسم الصورة بدون اللاحقة هو رقم اللوحة الصحيح تماماً (Ground Truth)
                true_plate_text = name_part.strip()
                
                # الصيغة القياسية لـ PaddleOCR: اسم_الصورة [زر الجدولة \t] النص_الصحيح
                line = f"{filename}\t{true_plate_text}\n"
                f.write(line)
                counter += 1
                
    print(f"✅ مبروك هندسة! تم توليد ملف الـ Labels بنجاح.")
    print(f"📊 إجمالي الصور التي تم فهرستها وتجهيزها: {counter} صورة.")
    print(f"📂 الملف الجاهز الآن هو: '{output_file}'")