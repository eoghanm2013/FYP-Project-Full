The product of this project is a web application found at: https://eoghanm.xyz/Activision/

There are two project folders, FYP & fypbackend.

FYP -----------> The Django Project folder contains all project files for the web application it uses
		 the data from fypbackend via a PostgreSQL database. This was pushed to DockerHub
		 and runs on a container on a cloud server.
		 FYP Folder has the general Django Configuration files
		 fypapp has all of the application files for the web application like model.p, 
		 views.py & the templates (HTML files).
		 CSS found in static/style.css





fypbackend-----> The backend processes are found in fypbackend. This is where the data collection
		 and the processing happens before the data is sent to the Database.
		 The scripts in this folder are ran using a task scheduler.
		 Twot app has the scraper.py which runs 24/7 get restarted once a day
		 AggData app analyser.py which runs once a day after 9pm
		 Predicitons app has predictions.py and runs once a day after scheduled an 90 mins after
		 anaylyser.py.