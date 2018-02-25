import logging

def calculate_performance(target, prediction):
    if len(prediction) != len(target):
        logging.info('Error: length different.')
        return

    # logging.info('target = {}'.format(target))
    # logging.info('prediction = {}'.format(prediction))
    logging.info('len(prediction) = {}'.format(len(prediction)))

    profit = 1
    profit_when_hold = 1
    for i in range(len(prediction)):
        if prediction[i] > 0:
            profit *= (1 + target[i])
        profit_when_hold *= (1 + target[i])
        # if i == 1:
        #     break

    a = 1
    for px in target:
        a *= (1 + px[0])
    logging.info('a = {}'.format(a))

    return profit, profit_when_hold