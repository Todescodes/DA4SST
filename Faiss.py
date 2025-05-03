# 必要なライブラリをインポート
import os
import json
import sys
import faiss
import csv
from tqdm import tqdm
import time
import pandas as pd
from pprint import pprint

# LangChainのテキスト分割器とローダーをインポート
from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import TextLoader

# 埋め込み用のHuggingFaceモデルをインポート
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# セマンティック検索用のデータベースクラス
class SemanticDB():
    '''
    このクラスは以下の5つの主要メソッドを含みます：
    1. prepare_database_runtime() : データベースのランタイムを初期化する（検索用）
    2. json_data_loader() : JSONファイルの読み込み
    3. csv_data_loader() : CSVファイルの読み込み
    4. create_faiss_db() : FAISSデータベースの作成
    5. search_db() : クエリ検索の実行
    '''

    def __init__(self, 
                 file_path, 
                 data_format,
                 db_path, 
                 content_key,
                 result_path,
                 result_data,
                 temp_query,
                 index,
                 input_index
                 ):
        '''
        コンストラクタ：
        - file_path : データファイルのパス
        - data_format : データの形式（json / csv / txt）
        - db_path : 作成したFAISS DBの保存先パス
        - content_key : データ中の検索対象コンテンツが格納されているキー名
        '''

        # HuggingFaceの事前学習済み埋め込みモデルを使ってベクトルを作成
        self.embeddings = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
            model_kwargs = {'device':'cuda'},
            encode_kwargs = {'normalize_embeddings': False}
        )

        self.file_path = file_path
        self.content_key = content_key
        self.db_path = db_path
        self.data_format = data_format

        self.result_path = result_path
        self.result_data = result_data
        self.temp_query = temp_query
        self.index = index
        self.input_index = input_index

        self.database_runtime = None
        
        # データベースが存在しない場合は作成
        if not os.path.exists(self.db_path):
            print("Database is not exist | Creating DB")
            self.create_faiss_db(
                file_path = self.file_path, 
                content_key = self.content_key, 
                db_path = self.db_path
            )

    def prepare_database_runtime(self):
        '''
        FAISSの検索用データベースを初期化
        '''
        db_runtime = FAISS.load_local(
            self.db_path, 
            self.embeddings, 
            allow_dangerous_deserialization = True
        )

        # GPUにインデックスを転送
        res = faiss.StandardGpuResources()
        db_runtime.index = faiss.index_cpu_to_gpu(res, 0, db_runtime.index)

        return db_runtime

    def txt_data_loader(self, file_path):
        # テキストファイルの読み込みとチャンク分割
        loader = TextLoader(file_path)
        data = loader.load()
        text_splitter = CharacterTextSplitter(separator= ".\n", chunk_size=1000, chunk_overlap=0, is_separator_regex=True)
        data = text_splitter.split_documents(data)
        print(data)
        sys.exit()
        return data


    def json_data_loader(self, file_path, content_key):
        '''
        JSONファイルを読み込んで、指定されたキーの内容をベースに文書リストを作成
        '''
        loader = JSONLoader(
            file_path = file_path,
            jq_schema = '.[]',
            text_content = False,
            json_lines = False,
            content_key = content_key,
        )
        data = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        data = text_splitter.split_documents(data)
        return data

    def csv_data_loader(self, file_path, content_key):
        '''
        CSVファイルの読み込みとチャンク分割
        '''
        loader = CSVLoader(
            file_path = file_path,
            source_column = content_key,    
        )
        data = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        data = text_splitter.split_documents(data)
        return data

    def create_faiss_db(self, file_path, content_key, db_path):
        '''
        指定されたファイル形式に応じてデータを読み込み、
        FAISSのインデックス（HNSW）を作成して保存する
        '''
        if self.data_format == "txt":
            data = self.txt_data_loader(file_path = file_path)
        elif self.data_format == "json":
            data = self.json_data_loader(file_path = file_path, content_key = content_key)
        elif self.data_format == "csv":
            data = self.csv_data_loader(file_path = file_path, content_key = content_key)
        else:
            raise "Unknown data format"

        # 文書からFAISSデータベースを作成
        db = FAISS.from_documents(data, self.embeddings)

        # HNSWインデックスの構築
        d = db.index.d  # 次元数
        M = 32  # ノード間のリンク数
        index = faiss.IndexHNSWFlat(d, M)
        
        vectors = db.index.xb
        index.add(vectors)

        # GPU上にインデックスを転送して保存
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        faiss.write_index(index, db_path)

    def search_db(self, query):
        '''
        データベース上でセマンティック検索を実行
        - query: クエリ文（検索したいテキスト）
        '''
        if self.database_runtime == None:
            self.evidence_db = self.prepare_database_runtime()
            self.database_runtime = self.prepare_database_runtime()

        # fetch_k: 類似度が高そうな候補を多めに検索
        # k: その中から最終的に返す数
        docs = self.evidence_db.similarity_search(query, k = 30, fetch_k = 100)
        return docs

    def csv_newdata(self, result_path, result_data, temp_query, index, input_index):
        '''
        類似検索結果をCSVファイルに追記して記録する処理
        - input: negativeの入力文
        - reference: positiveな検索結果
        '''
        if os.path.isfile('HNSW_data.csv'):
            with open('HNSW_data.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([result_data,"negative",index,temp_query,"positive",input_index])
        else:
            with open('HNSW_data.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["input","i_sentiment","I_location","reference","r_sentiment","r_location","semantice_similarity"])
                writer.writerow([result_data,"negative",index,temp_query,"positive",input_index])


# メイン処理：negative（0）をDB化 → positive（1）文とのペアを作成
if __name__ == "__main__":
    
    #negativeなテキストのファイルパスを指定
    raw_file_dir = "datasets/yelp/sentiment.train.0"

    with open(raw_file_dir, "r") as txt_reader:
        database = [{"semantic": d.replace('\n', '')} for d in txt_reader.readlines()]

    with open(f"{raw_file_dir}.json", "w") as json_w:
        json.dump(database, json_w, indent = 4)
    
    
    langchain_json = SemanticDB(
        file_path = f"{raw_file_dir}.json",
        data_format = "json",
        db_path = "database/yelp_negative", 
        content_key = "semantic",
        result_path = "data",
        result_data = "",
        temp_query = "",
        index = "",
        input_index = ""
    )

    #positiveなテキストのファイルパスを指定
    with open("datasets/yelp/sentiment.train.1") as f :
        n=1
        iterations = 267314

        for line in tqdm(f, total=iterations, desc="ペア作成中"):
            input_text = line.strip()
            if not input_text:  # 空行を無視
                continue
                
            # you can change the query as you want.
            all_search_result = langchain_json.search_db(query = input_text)

            langchain_json.temp_query = input_text

            for results in all_search_result:
                langchain_json.csv_newdata(result_path = "data",result_data = results.page_content,temp_query = input_text ,index = results.metadata["seq_num"],input_index = n )

            n += 1
            
            time.sleep(0.0001)

