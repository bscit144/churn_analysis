# churn_analysis
-first we have to import libraries necessary for logistics regression.then load the data using the pandas library.
-We have explore the data by checking whether there is null values,the datatypes that makes the data,check the number of columns androws the data has,
-After exploring the data,we perform data wrangling where we replace null values
-Through analysis,we find that ,there is large number of males captured in the dataset in all the three geographical countries that female.
The analysis also shows;
        large number of females exited as compare to their male counterparts
        from the show of boxplot,people from germany exited in large number as compared to those from other two countries captured in the dataset
-Since logistic regression only uses numerical data for prediction, we have converted the string values to categorical values inform of 1 and 0,in this 
case, the data for Geography,and Gender have been converted into 1 and 0 where for gender,1 rep male and 0 for females
