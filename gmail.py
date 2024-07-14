import pandas as pd
import email
from email import policy
from email.parser import BytesParser
from datetime import datetime
from sqlalchemy import create_engine, text, VARBINARY, Integer
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from transformers import BertTokenizer, BertModel
import torch
import pickle
import pyodbc
import binascii
import urllib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
from getpass import getpass
from langchain_community.embeddings import CohereEmbeddings
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import spacy
from langchain_community.llms import Cohere
from datetime import datetime
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nlp_date_parser import nlp_date_parser

class EmailSearch:
    def __init__(self, connection_string, COHERE_API_KEY, thr=0.65, D='auto'):
        self.connection_string = connection_string
        self.thr = thr
        if D == 'auto':
            self.D = thr - 1
        else:
            self.D = D
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./')
        self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir='./')
        os.environ["COHERE_API_KEY"] = COHERE_API_KEY
        self.llm = Cohere(temperature=0)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        # self.nlp = spacy.load("en_core_web_sm")
        self.query_template = """
        SELECT EmailId
        FROM Enron_emaildataset
        WHERE (:from_email IS NULL OR FromEmail LIKE '%' + :from_email + '%')
          AND (:to_email IS NULL OR ToEmail LIKE '%' + :to_email + '%')
          AND (:start_date IS NULL OR DateEmail >= :start_date)
          AND (:end_date IS NULL OR DateEmail <= :end_date)
        """
        self.cache = None
        self.date_parser = nlp_date_parser()
        
    def get_bert_embeddings(self,tokens_list):
        # Convert tokens list to tensor
        tokens_tensor = torch.tensor(tokens_list)
    
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            embeddings = outputs.last_hidden_state
        return embeddings

    def get_engine(self):
        params = urllib.parse.quote_plus(self.connection_string)
        self.engine = create_engine(f'mssql+pyodbc:///?odbc_connect={params}')
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                print("Connection successful!")
                self.Session = sessionmaker(bind=self.engine)
                print('Session defined')
        except Exception as e:
            print(f"Connection failed: {e}")

    def parse_email_message(self,message):
        msg = email.message_from_string(message, policy=policy.default)
        email_data = {
            'FromEmail': msg.get('X-From'),
            'ToEmail': msg.get('X-To'),
            'DateEmail': msg.get('Date'),
            'SubjectEmail': msg.get('Subject'),
            'BodyEmail': msg.get_payload(decode=True).decode('us-ascii', errors='ignore') if msg.is_multipart() else msg.get_payload()
        }
        return email_data

    def create_table(self,table_name,drop_table_query,create_table_query):
        session = self.Session()
        try:
            session.execute(text(drop_table_query))
            session.commit()
            session.execute(text(create_table_query))
            session.commit()
            print(f"Table '{table_name}' created successfully.")
        except SQLAlchemyError as e:
            session.rollback()
            print(f"Error occurred while creating the table: {e}")
        finally:
            session.close()

    def define_database(self):
        table_name = 'Enron_emaildataset'
        drop_table_query = f"""
        DROP TABLE IF EXISTS {table_name}
        """
        create_table_query = f"""
            CREATE TABLE {table_name} ( 
                [FromEmail] varchar(100),
                [ToEmail] varchar(MAX),
                [DateEmail] datetime,
                [SubjectEmail] varchar(MAX),
                [BodyEmail] nvarchar(MAX),
                [EmailId] bigint  
            )
        """
        self.create_table(table_name,drop_table_query,create_table_query)

        table_name = 'subject_embedding_data'
        drop_table_query = f"""
        DROP TABLE IF EXISTS {table_name}
        """
        
        create_table_query = f"""
        CREATE TABLE {table_name} (
            EmailId bigint,
            Subject_embeddings_bin VARBINARY(MAX)
        )
        """
        self.create_table(table_name,drop_table_query,create_table_query)

        table_name = 'body_embedding_data'
        drop_table_query = f"""
        DROP TABLE IF EXISTS {table_name}
        """
        
        create_table_query = f"""
        CREATE TABLE {table_name} (
            EmailId bigint,
            Body_embeddings_bin VARBINARY(MAX)
        )
        """
        self.create_table(table_name,drop_table_query,create_table_query)

    def Create_database(self,csv_file_path):
        self.define_database()
        df = pd.read_csv(csv_file_path)
        # Apply the function to the 'message' column and create a new DataFrame
        parsed_emails = df['message'].apply(self.parse_email_message)
        
        # Convert the parsed emails into a DataFrame
        parsed_emails_df = pd.DataFrame(parsed_emails.tolist())
        
        result_df = pd.concat([df['file'], parsed_emails_df], axis=1)
        result_df['DateEmail'] = pd.to_datetime(result_df['DateEmail']).dt.tz_localize(None)        
        result_df['EmailId'] = range(1, len(result_df) + 1)
        result_df['SubjectEmail'] = result_df['SubjectEmail'].str.lower()
        result_df['BodyEmail'] = result_df['BodyEmail'].str.lower()
        result_df = result_df.drop(columns=['file'])

        result_df.to_sql('Enron_emaildataset', con=self.engine, if_exists='append', index=False)

        result_df['Subject_tokens'] = result_df['SubjectEmail'].apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True, max_length=512, truncation=True))
        result_df['Body_tokens'] = result_df['BodyEmail'].apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True, max_length=512, truncation=True))
        
        # Get embeddings for 'Subject' and 'Body' tokens
        result_df['Subject_embeddings'] = result_df['Subject_tokens'].apply(lambda x: self.get_bert_embeddings([x])[0])
        result_df['Body_embeddings'] = result_df['Body_tokens'].apply(lambda x: self.get_bert_embeddings([x])[0])
        result_df['Subject_embeddings'] = result_df['Subject_embeddings'].apply(lambda x: x.tolist())
        result_df['Body_embeddings'] = result_df['Body_embeddings'].apply(lambda x: x.tolist())
        # Convert embeddings to VARBINARY format using pickle.dumps()
        result_df['Subject_embeddings_bin'] = result_df['Subject_embeddings'].apply(lambda x: pickle.dumps(x))
        result_df['Body_embeddings_bin'] = result_df['Body_embeddings'].apply(lambda x: pickle.dumps(x))
        
        subject_df = result_df[['EmailId','Subject_embeddings_bin']]
        Body_df = result_df[['EmailId','Body_embeddings_bin']]

        subject_df.to_sql('subject_embedding_data', con=self.engine, if_exists='append', index=False, dtype={
            'Subject_embeddings_bin': VARBINARY(length='max'),  # Use SQLAlchemy's VARBINARY type
            'EmailId': Integer  # Ensure EmailId is mapped to Integer if not already
        })

        Body_df.to_sql('body_embedding_data', self.engine, if_exists='append', index=False, dtype={
            'Body_embeddings_bin': VARBINARY(length='max'),  # Use SQLAlchemy's VARBINARY type
            'EmailId': Integer  # Ensure EmailId is mapped to Integer if not already
        })

        return self

    def process_text(self,text):
        # Tokenization
        tokens = word_tokenize(text)
            
        # Stopword removal
        tokens = [word for word in tokens if word.lower() not in self.stop_words]
            
        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            
        return tokens

    def get_query_embeddings(self,query):
        query = query.lower()
        query_tokens = self.tokenizer.encode(query, add_special_tokens=True, max_length=512, truncation=True)
        query_embedding = self.get_bert_embeddings([query_tokens])[0]
        return query_embedding

    def convert_natural_language_to_mail_search(self,query):
        # Define the prompt for the language model
        prompt = prompt = f"""
        Convert the following natural language query to a mail search query format.
    
        Query: "{query}"
    
        Format:
        From : <Sender's name>
        To : <Recipient's name>
        Date : <Date or date range>
        Subject : <Subject of the email>
    
        If any information is not present in the query, return None for that field.
    
        Example:
        Query: "Find emails from John to Jane and Bill W. last week about the project update"
        Output:
        From : John
        To : [Jane, Bill W.]
        Date : last week
        Subject : project update
    
        Example:
        Query: "Show me the emails sent by Alice regarding the budget report"
        Output:
        From : Alice
        To : None
        Date : None
        Subject : budget report

        Example:
        Query: "mail received in may 2001"
        Output:
        From : None
        To : None
        Date : may 2001
        Subject : None

        Example:
        Query: "Show me the emails sent by Alice on 5 May 2002 regarding the budget report"
        Output:
        From : Alice
        To : None
        Date : 5 May 2002
        Subject : budget report
    
        Query: "{query}"
        Output:
        """
    
        # Use the language model to generate the output
        output = self.llm(prompt)
    
        # Parse the output into the required format
        lines = output.strip().split('\n')
        result = {}
        for line in lines:
            if ': ' in line:
                key, value = line.split(': ', 1)
                result[key.strip()] = value.strip() if value.strip() else None
    
        return result

    def Get_score(self,similarity_score, thr, D):
        similarity_score[similarity_score < thr] = 0
        cache_score = similarity_score
        cache_score[similarity_score >= thr] = 1
        self.cache += np.sum(cache_score, axis=0)
        token_wise_score = np.sum(similarity_score, axis=0)
        token_wise_score[token_wise_score == 0] = -D
        return token_wise_score

    def final_score(self,token_wise_score):
        self.cache[self.cache == 0 ] = 1
        score = token_wise_score/self.cache
        return np.sum(score)
    
    def search(self, query):
        mail_search_query = self.convert_natural_language_to_mail_search(query)
        for key in ['From', 'To', 'Date', 'Subject']:
            if key not in mail_search_query.keys():
                mail_search_query[key] = None
                
        if mail_search_query['Date'] == 'None':
            date = {'start_date':None,'end_date':None}
        else:
            date = self.date_parser(mail_search_query['Date'])
        if date == None:
            date = {'start_date':None,'end_date':None}

        params = {
            'from_email': mail_search_query['From'],
            'to_email': mail_search_query['To'],
            'start_date': date['start_date'],
            'end_date': date['end_date']
        }
        # Adjust params to None where needed
        for key in params:
            if params[key] == 'None':
                params[key] = None

        if mail_search_query['Subject'] == 'None':
            query_template = """
                SELECT EmailId 
                FROM Enron_emaildataset
                WHERE (:from_email IS NULL OR FromEmail LIKE '%' + :from_email + '%')
                  AND (:to_email IS NULL OR ToEmail LIKE '%' + :to_email + '%')
                  AND (:start_date IS NULL OR DateEmail >= :start_date)
                  AND (:end_date IS NULL OR DateEmail <= :end_date)
                """
            with self.engine.connect() as conn:
                result = conn.execute(text(query_template), params)
                emails_info = result.fetchall()

            emails_info = tuple(id[0] for id in emails_info)
            if not emails_info:
                return tuple()  # No matching emails

            return emails_info

        # Execute the query
        with self.engine.connect() as conn:
            result = conn.execute(text(self.query_template), params)
            emails_info = result.fetchall()

        emails_info = tuple(id[0] for id in emails_info)
        if not emails_info:
            return tuple()  # No matching emails

        table_name = 'body_embedding_data'
        embedding_column = 'EmailId, Body_embeddings_bin'
        query = f"SELECT {embedding_column} FROM {table_name} WHERE EmailId in {emails_info}"
        query_dfb = pd.read_sql(query, self.engine)
        table_name = 'subject_embedding_data'
        embedding_column = 'EmailId, Subject_embeddings_bin'
        query = f"SELECT {embedding_column} FROM {table_name} WHERE EmailId in {emails_info}"
        query_dfs = pd.read_sql(query, self.engine)
        query_df = pd.merge(query_dfs, query_dfb, on='EmailId')
        query_df['Subject_embeddings_bin'] = query_df['Subject_embeddings_bin'].apply(lambda x: pickle.loads(x))
        query_df['Body_embeddings_bin'] = query_df['Body_embeddings_bin'].apply(lambda x: pickle.loads(x))

        Subject_query = mail_search_query['Subject']
        processed_tokens = self.process_text(Subject_query)
        processed_query = ' '.join(processed_tokens)
        query_embedding = self.get_query_embeddings(processed_query)
        query_df['Subject_similarity'] = query_df['Subject_embeddings_bin'].apply(lambda x: cosine_similarity(x, query_embedding))
        query_df['Body_similarity'] = query_df['Body_embeddings_bin'].apply(lambda x: cosine_similarity(x, query_embedding))

        self.cache = np.zeros(len(query_embedding))
        query_df['Subject_similarity_score'] = query_df['Subject_similarity'].apply(lambda x: self.Get_score(x, self.thr, self.D))
        query_df['Subject_similarity_score'] = query_df['Subject_similarity'].apply(lambda x: self.final_score(x))
        self.cache = np.zeros(len(query_embedding))
        query_df['Body_similarity_score'] = query_df['Body_similarity'].apply(lambda x: self.Get_score(x, self.thr, self.D))
        query_df['Body_similarity_score'] = query_df['Body_similarity'].apply(lambda x: self.final_score(x))

        query_df['score'] = query_df['Subject_similarity_score'] + query_df['Body_similarity_score']
        query_df = query_df.sort_values(by='score', ascending=False)

        return tuple(query_df['EmailId'].to_list())

    def get_mail(self,id):
        query = f"SELECT EmailId, FromEmail, ToEmail, DateEmail, SubjectEmail, BodyEmail FROM Enron_emaildataset WHERE EmailId = {id}"
        df = pd.read_sql(query, self.engine)
        return df

    def get_mails(self,ids):
        if type(ids) == int:
            df = self.get_mail(ids)
        else:
            query = f"SELECT EmailId, FromEmail, ToEmail, DateEmail, SubjectEmail, BodyEmail FROM Enron_emaildataset WHERE EmailId in {ids}"
            df = pd.read_sql(query, self.engine)
            df = df.set_index('EmailId')
            df = df.reindex(ids).reset_index()
        return df