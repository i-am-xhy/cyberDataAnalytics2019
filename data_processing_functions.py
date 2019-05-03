def process_dict_reader(dictreader):
    dictlist = remove_refused(dictreader)
    dictlist = clean_formatting(dictlist)
    dictlist = add_imf_currency_conversion(dictlist)
    return dictlist


def remove_refused(dictlist):
    result = []
    for line in dictlist:
        if line['simple_journal'] == 'refused':
            continue
        if line['bin'] == 'na' or line['mail_id'] == 'na':
            continue

        result.append(line)

    return result


def clean_formatting(dictlist):
    for line in dictlist:
        # chargebacks can be assumed to be fraud, whereas the rest should be safe.
        if line['simple_journal'] == 'Chargeback':
            line['label'] = 1 # fraud
        else:
            line['label'] = 0 # safe

        # 0 = Unknown, 1=Match, 2=No Match, 3-6=Not checked
        line['cvcresponsecode'] = int(line['cvcresponsecode'])
        if line['cvcresponsecode'] > 2:
            line['cvcresponsecode'] = 3

    return dictlist
# how many of that currency is worth 1 XDR or imf special drawing right.
xdr_conversion_rates = {
    'SEK': 13.2510,
    'NZD': 2.09126,
    'GBP': 1.97804,
    'MXN': 26.4705,
    'AUD': 1.97804

}
def add_imf_currency_conversion(dictlist):
    # adds an additional currency field in which the base currency is transferred to a uniform currency
    # uses imf special drawing rights to be more stable, might need to be expanded to take into account relative purchase power over time
    for line in dictlist:
        currency = line['currencycode']
        amount = float(line['amount'])

        line['xdr_amount'] = amount/xdr_conversion_rates[currency]
    return dictlist

def get_distinct_in_column(dictlist, column_name):
    # prints all distinct values in a given column
    result = set()
    for line in dictlist:
        result.add(line[column_name])
    return result

def split_label(dictlist):
    safe_result = []
    fraud_result = []
    for line in dictlist:
        if line['label'] == 0:
            safe_result.append(line)
        else:
            fraud_result.append(line)
    return safe_result, fraud_result

def get_column(dictlist, column_name):
    # returns a column as a list
    result = []
    for line in dictlist:
        result.append(line[column_name])
    return result
