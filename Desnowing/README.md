### Download the Datasets

- CSD [[gdrive](https://drive.google.com/file/d/1pns-7uWy-0SamxjA40qOCkkhSu7o7ULb/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1N52Jnx0co9udJeYrbd3blA?pwd=sb4a)]

### Training

~~~
python main.py --mode train --data_dir your_path/CSD
~~~

### Evaluation
#### Testing
~~~
python main.py --data_dir your_path/CSD --test_model path_to_csd_model
~~~

For training and testing, your directory structure should look like this

`Your path` <br/>
 `├──CSD` <br/>
     `├──train2500`  <br/>
          `├──Gt`  <br/>
          `└──Snow`  
     `└──test2000`  <br/>
          `├──Gt`  <br/>
          `└──Snow`  
