# technical_test

localhost:5000/models

{
    "models": [
        "lightgbm_output_v0.1-1588759220.335498"
    ]
}


localhost:5000/output/lightgbm_output_v0.1-1588759220.335498
raw data :

{
	"checking_balance": "< 0 DM",
	"months_loan_duration": 47,
	"credit_history": "repaid",
	"purpose": "car (new)",
	"amount": 10722,
	"savings_balance": "< 100 DM",
	"employment_length": "0 - 1 yrs",
	"installment_rate": 1,
	"personal_status": "female",
	"other_debtors": "none",
	"residence_history": 1,
	"property": "real estate",
	"age": 35,
	"installment_plan": "none",
	"housing": "own",
	"existing_credits": 1,
	"dependents": 1,
	"telephone": "yes",
	"foreign_worker": "yes",
	"job": "unskilled resident"
	
}

localhost:5000/predict
default value





localhost:5000/outputs/lightgbm_output_v0.1-1588759220.335498
qui load le dataset test.csv

localhost:5000/outputs_upload/lightgbm_output_v0.1-1588759220.335498
on choisit le csv
form-data
fileupload : test.csv


Projet de mise en production d'un algorithme pour la prédiction d'embauche pour le poste de chercheur d'or chez Orfée.


# Installation en local sans docker
create a virtual env :
    
    python3 -m venv embauche_venv

Activate the virtual env
on windows :


    embauche_venv\Scripts\activate.bat

on linux or macos :


    source embauche_venv/bin/activate
    
Install dependencies :

    pip install -r requirements.txt
    
Pour sauvegarder les dépendances :

    pip freeze > requirements.txt
    
    

# Installation avec Docker :

    docker build -t nguyenanht/fairmoney:latest .
    docker run -p 5000:5000 -it nguyenanht/fairmoney:latest
    
    
    
Elastic beanstalk deploy:

init beanstalk :

    eb init -p docker application-name 
    
test locally :
    
    eb local run --port 5000
    
    
    eb create application-name
    
    
https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/single-container-docker.html
https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/single-container-docker.html

    
# Pour lancer le programme en local :

### Avec Docker :

    docker run -it nguyenanht/hiring:latest
  
### Avec une virtuel environnement python :

Pour réentrainer un modèle :


    python3 src/training_pipeline 

Pour lancer le flask :


    python3 application.py
    
