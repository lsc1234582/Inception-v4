wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz;
tar xvf 101_ObjectCategories.tar.gz;
mv 101_ObjectCategories raw_images;
mkdir bin;
python convert_image_to_structs.py raw_images bin;
rm 101_ObjectCategories.tar.gz
