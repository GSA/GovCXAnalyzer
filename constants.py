# oa11 = {'Domain': ['Satisfaction', 'Trust/Confidence',
#                                      'Quality', 'Ease/Simplicity', 'Efficiency/Speed',
#                                      'Equity/Transparency', "Helpfulness"],
                                     
                                     
                                     
                                    
                
#               'Question': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6','Q7',# 'Q8'
#                           ],
#               'Prompt': [
#         'I am satisfied with the information I received from [service]',
#        'This interaction increased my confidence in [service]',
#        'I was able to easily find what I was looking for.',
#        'The information was useful.', 

#           'The website was easy to navigate.' ,
#            'It took a reasonable amount of time to do what I needed to do.'
#                   ,
#               "Helpful"]
#                                      }


idcols =  ['ID', 'UUID']

additional_cols = ['Created At', "Page", "Referrer", "User Agent"]

drivers2questions = {
'Satisfaction': 'I am satisfied with the service I received from [service]',
 'Trust/Confidence': 'This interaction increased my confidence in [service provider]',
 'Effectiveness/Quality': 'My need was addressed.',
  'UserGroup': "Which best describes you?",
 "Years": "How many years have you been enrolled?",
 'Region': "Which region are you in?",
 'Ease/Simplicity': 'It was easy to complete what I needed to do.',
 'Efficiency/Speed': 'It took a reasonable amount of time to do what I needed to do.',
 'Equity/Transparency': "I understood what was being asked of me throughout the process.",
 'Helpfulness': 'The [service] helped me do what I needed to do.',
 "FreeText": "Please share any additional feedback you have about your experience with [service].",
 }
