import os
from typing import Dict
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter  # Pour améliorer l'image
from dotenv import load_dotenv

class CVDataExtractor:
    def __init__(self, openai_api_key: str):
        self.llm = OpenAI(api_key=openai_api_key, temperature=0)

        self.extract_prompt = PromptTemplate(
            input_variables=["cv_text"],
            template="""
            Tu es un extracteur de données de CV. 
            Retourne exclusivement un objet JSON valide avec ces champs : 

            {{
                "prenom": "valeur ou chaine vide",
                "nom": "valeur ou chaine vide",
                "email": "valeur ou chaine vide",
                "telephone": "valeur ou chaine vide",
                "code_postal": "valeur ou chaine vide",
                "ville": "valeur ou chaine vide",
                "recherche": "valeur ou chaine vide"
            }}

            Analyse le texte suivant et extrais les informations demandées :
            ```text
            {cv_text}
            ```
            """
        )

        self.extraction_chain = LLMChain(llm=self.llm, prompt=self.extract_prompt)
        self.cv_index = 0  

    def _clean_and_validate_data(self, data: Dict) -> Dict:
        cleaned_data = {'index': self.cv_index}
        self.cv_index += 1  

        fields = ["prenom", "nom", "email", "telephone", "code_postal", "ville", "recherche"]
        for field in fields:
            cleaned_data[field] = data.get(field, "")

        phone = cleaned_data['telephone']
        if phone:
            phone = phone.replace(' ', '').replace('.', '')  
            if phone.startswith('0'):
                phone = '+33' + phone[1:]  
            if phone.startswith('+33'):
                rest = phone[3:]
                cleaned_data['telephone'] = '+33 ' + ' '.join(rest[i:i+2] for i in range(0, len(rest), 2))
            else:
                cleaned_data['telephone'] = ' '.join(phone[i:i+2] for i in range(0, len(phone), 2))

        return cleaned_data

    def improve_image(self, img: Image) -> Image:
        """
        Applique des améliorations à l'image avant l'OCR.
        - Convertir en niveaux de gris
        - Augmenter le contraste
        - Appliquer un filtre de netteté
        """
        # Convertir l'image en niveaux de gris
        img = img.convert('L')
        # Améliorer le contraste
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2)  # Augmenter le contraste de 2x
        # Appliquer un filtre de netteté
        img = img.filter(ImageFilter.SHARPEN)
        return img

    def process_pdf(self, pdf_path: str, manual_page_check: list = None) -> Dict:
        all_results = []
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            full_text = " ".join([page.page_content for page in pages])

            if full_text.strip():
                all_results.extend(self._process_text_in_chunks(full_text, pdf_path))

            images = convert_from_path(pdf_path)
            for i, img in enumerate(images):
                # Si l'utilisateur veut vérifier manuellement cette page
                if manual_page_check and i+1 in manual_page_check:
                    print(f"Vérification manuelle de la page {i+1} :")
                    img.show()  # Afficher l'image de la page pour la vérification
                    page_text = pytesseract.image_to_string(img, lang='fra+eng')
                    print(f"Texte OCR pour la page {i+1} :\n{page_text}\n")
                # Améliorer l'image avant de l'envoyer à l'OCR
                img = self.improve_image(img)
                ocr_text = pytesseract.image_to_string(img, lang='fra+eng')
                print(f"OCR Text extrait de la page {i+1} :\n{ocr_text}\n")
                if ocr_text.strip():
                    all_results.extend(self._process_text_in_chunks(ocr_text, pdf_path))

        except Exception as e:
            print(f"❌ Erreur lors du traitement de {pdf_path}: {str(e)}")
        
        return all_results

    def _process_text_in_chunks(self, text: str, pdf_path: str) -> Dict:
        max_tokens = 4096
        chunk_size = max_tokens - 256
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        chunk_results = []
        for chunk in chunks:
            try:
                result = self.extraction_chain.invoke({"cv_text": chunk})
                
                if not result or 'text' not in result or not result['text'].strip():
                    print(f"⚠️ Aucune réponse de l'LLM pour {pdf_path}.")
                    continue

                print(f"🔍 Réponse brute de l'LLM pour {pdf_path} :\n{result['text']}")

                extracted_data = json.loads(result['text'])
                cleaned_data = self._clean_and_validate_data(extracted_data)
                cleaned_data['file_name'] = os.path.basename(pdf_path)
                chunk_results.append(cleaned_data)

            except json.JSONDecodeError as e:
                print(f"❌ Erreur de décodage JSON pour {pdf_path}: {str(e)}")
                continue
            except Exception as e:
                print(f"❌ Erreur inattendue pour {pdf_path}: {str(e)}")
                continue

        return chunk_results

    def process_directory(self, path: str, manual_page_check: list = None) -> pd.DataFrame:
        results = []

        if os.path.isdir(path):
            self.cv_index = 0
            for filename in os.listdir(path):
                if filename.lower().endswith('.pdf'):
                    file_path = os.path.join(path, filename)
                    file_results = self.process_pdf(file_path, manual_page_check)
                    results.extend(file_results)
        elif os.path.isfile(path) and path.lower().endswith('.pdf'):
            self.cv_index = 0
            file_results = self.process_pdf(path, manual_page_check)
            results.extend(file_results)
        else:
            print(f"❌ Le chemin {path} n'est ni un dossier ni un fichier PDF valide.")

        return pd.DataFrame(results) if results else pd.DataFrame()

    def save_results(self, df: pd.DataFrame, output_path: str):
        if df.empty:
            print("⚠️ Aucune donnée à sauvegarder")
            return

        columns_order = ['index', 'nom', 'prenom', 'email', 'telephone', 'code_postal', 'ville', 'recherche', 'file_name']
        for col in columns_order:
            if col not in df.columns:
                df[col] = ""
        df = df[columns_order]

        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n✅ Données sauvegardées dans : {output_path}")

def main():
    load_dotenv()
    extractor = CVDataExtractor(os.getenv('OPENAI_API_KEY'))

    input_path = "/Users/lea/Desktop/CV_Salon/Web_Dev_ou_plus.pdf"
    output_path = "resultats_cv.csv"
    
    # Exemple : vérifier manuellement les pages 1 et 3
    # manual_page_check = [ ]  # Ajouter les pages que tu veux vérifier manuellement

    results_df = extractor.process_directory(input_path)

    if not results_df.empty:
        extractor.save_results(results_df, output_path)
        print("\n📊 Résultats extraits:")
        print(results_df)
    else:
        print("❌ Aucune donnée extraite.")

if __name__ == "__main__":
    main()
