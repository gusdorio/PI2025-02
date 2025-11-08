# Presentation (portuguese)

Este é um Projeto Integrador relativo ao Segundo Semestre 2025, da Escola Politecnica PUC Campinas. Tem como objetivo desenvolver uma aplicação que reúne conceitos de 3 matérias distintas (Computação em Nuvem, Transformação de Dados e Aprendizado Supervisionado)

# Project Structure

This repository is separated into a two side microsservice applications, with one side responsible for user operations and the another for back-end operations. Details below.

## Microsservice 1 (dashboard app)
'dashboard/' works as a user interactive web application, where the user is able to:

- upload data files (which are passed to our another one service);
- verify results from data imports after data processing operations executed by the another service.

The data operations occurs essentially by adopting a client interface, which passes the data via http POST requests to the another microsservice in a pre-defined route; and by accesing a mongodb connection, to pull results from what is done in the another microsservice data operation tasks.

## Microsservice 2 (machine learning model app)
'ml_model' works as an application responsible, in essence, to execute an machine learning algorithm to mine results from data files imported at the dashboard microsservice side. This process happens by:

- a simple http server is started at the main entry; it listens for post calls from dashboard side, expecting to receive a special data object crafted from 'models' folder;
- special data operations are called to verify the data, transform it, and then passes to the most important step, which is to do the computations related to the machine learning algorithm itself;
- at the end of the algorithm execution, the results are stored at the mongodb connection.


## Special Considerations
The 'models/' folder is intended to serve the two sides of microsservice applications, in order to deliver the necessary resources which makes the services able to stablish the connection with the database used in this project (MongoDB). It also have some common operations related to data transformation operations which receive transformations in the dashboard service side and are re-transformed to original again at the ml_model service side.