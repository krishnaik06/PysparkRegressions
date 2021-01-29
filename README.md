# Pyspark ML
Pyspark

Using pipenv to create virtual enviorment which content requirements



- install pipenv
  If you don't have pipenv you need to install it you can see detail [here](https://pipenv-fork.readthedocs.io/en/latest/)

- fork this repo
    ![alt text](./img/Screenshot%202020-11-09%20at%2015.41.07.png "screen")
- clone this repo in the your system
  ```bash
  git clone https://github.com/piyushtada/Pyspark-ML.git
  ```
- then change directory to Pyspark-ML
  ```bash
  cd Pyspark-ML
  ```
- Then run command pipenv install
  ```bash
  pipenv install
  ```
  this will install all the dependences you need to run the project
- Run jupyter notebook
  ```bash
  jupyter notebook
  ```
  it will open the jupyter notebook and you can use spark in it.
- Check if everything is working by using test.ipynb 

- when you want to open the secission again you need run following command after going in the PysparkML folder
    ```bash
  pipenv shell  
  jupyter notebook
  ```


# List of tasks in the project
- [x] Do exploratory data analysis
- [x] Make update to columns with categorical data
- [x] Visualise the results
- [x] Make data ready for models 
- [ ] Save the file 
- [ ] Run one sample model to check everything uptill now working
- [ ] Make list of models to apply
- [ ] Apply models
- [ ] Do hyperparameter tuning for the model