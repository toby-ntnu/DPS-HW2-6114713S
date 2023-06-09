import importlib,sys
importlib.reload(sys)
import pandas as pd
 
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
 
DB_CONNECT_STRING = 'mysql+pymysql://root:0009@localhost:3306/hw?charset=utf8'
engine = create_engine(DB_CONNECT_STRING, echo=True)
DB_Session = sessionmaker(bind=engine)
session = DB_Session()
 
data1 = pd.read_csv('newname_107.csv')
print(data1.shape)
pd.io.sql.to_sql(frame=data1,name='name107',con=engine,index=False,if_exists='replace')
