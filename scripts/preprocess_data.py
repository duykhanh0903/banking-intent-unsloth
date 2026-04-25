import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import os
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s?.,!']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_data(df_train, df_test, df_val):

    df_train['text'] = df_train['text'].apply(clean_text)
    df_test['text'] = df_test['text'].apply(clean_text)
    df_val['text'] = df_val['text'].apply(clean_text)

    return df_train, df_test, df_val

def mapping_label(df):
    label = {0: "activate_my_card",
             1: "age_limit",
             2: "apple_pay_or_google_pay",
             3: "atm_support",
             4: "automatic_top_up",
             5: "balance_not_updated_after_bank_transfer",
             6: "balance_not_updated_after_cheque_or_cash_deposit",
             7: "beneficiary_not_allowed",
             8: "cancel_transfer",
             9: "card_about_to_expire",
             10: "card_acceptance",
             11: "card_arrival",
             12: "card_delivery_estimate",
             13: "card_linking",
             14: "card_not_working",
             15: "card_payment_fee_charged",
             16: "card_payment_not_recognised",
             17: "card_payment_wrong_exchange_rate",
             18: "card_swallowed",
             19: "cash_withdrawal_charge",
             20: "cash_withdrawal_not_recognised",
             21: "change_pin",
             22: "compromised_card",
             23: "contactless_not_working",
             24: "country_support",
             25: "declined_card_payment",
             26: "declined_cash_withdrawal",
             27: "declined_transfer",
             28: "direct_debit_payment_not_recognised",
             29: "disposable_card_limits",
             30: "edit_personal_details",
             31: "exchange_charge",
             32: "exchange_rate",
             33: "exchange_via_app",
             34: "extra_charge_on_statement",
             35: "failed_transfer",
             36: "fiat_currency_support",
             37: "get_disposable_virtual_card",
             38: "get_physical_card",
             39: "getting_spare_card",
             40: "getting_virtual_card",
             41: "lost_or_stolen_card",
             42: "lost_or_stolen_phone",
             43: "order_physical_card",
             44: "passcode_forgotten",
             45: "pending_card_payment",
             46: "pending_cash_withdrawal",
             47: "pending_top_up",
             48: "pending_transfer",
             49: "pin_blocked",
             50: "receiving_money",
             51: "Refund_not_showing_up",
             52: "request_refund",
             53: "reverted_card_payment?",
             54: "supported_cards_and_currencies",
             55: "terminate_account",
             56: "top_up_by_bank_transfer_charge",
             57: "top_up_by_card_charge",
             58: "top_up_by_cash_or_cheque",
             59: "top_up_failed",
             60: "top_up_limits",
             61: "top_up_reverted",
             62: "topping_up_by_card",
             63: "transaction_charged_twice",
             64: "transfer_fee_charged",
             65: "transfer_into_account",
             66: "transfer_not_received_by_recipient",
             67: "transfer_timing",
             68: "unable_to_verify_identity",
             69: "verify_my_identity",
             70: "verify_source_of_funds",
             71: "verify_top_up",
             72: "virtual_card_not_working",
             73: "visa_or_mastercard",
             74: "why_verify_identity",
             75: "wrong_amount_of_cash_received",
             76: "wrong_exchange_rate_for_cash_withdrawal"
            }
    df['label'] = df['label'].map(label)
    return df

if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset("banking77")
    df_train = pd.DataFrame(dataset['train'])
    df_test = pd.DataFrame(dataset['test'])

    df_train_sampled, _ = train_test_split(
        df_train, 
        train_size=0.4, 
        stratify=df_train['label'], 
        random_state=42
    )

    df_test_sampled, _ = train_test_split(
        df_test, 
        train_size=0.15, 
        stratify=df_test['label'], 
        random_state=42
    )

    df_train_sampled, df_val_sampled = train_test_split(
        df_train_sampled, 
        test_size=0.1, 
        stratify=df_train_sampled['label'], 
        random_state=42
    )

    print(f"Đã lưu {len(df_train_sampled)} mẫu train, {len(df_test_sampled)} mẫu test và {len(df_val_sampled)} mẫu validation.")

    df_train, df_test, df_val = preprocess_data(df_train_sampled, df_test_sampled, df_val_sampled)

    df_train = mapping_label(df_train)
    df_test = mapping_label(df_test)
    df_val = mapping_label(df_val)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    data_path = os.path.join(parent_dir, "sample_data")

    os.makedirs(data_path, exist_ok=True)
    df_train.to_csv(os.path.join(data_path, "train.csv"), index=False)
    df_test.to_csv(os.path.join(data_path, "test.csv"), index=False)
    df_val.to_csv(os.path.join(data_path, "val.csv"), index=False)

    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    





