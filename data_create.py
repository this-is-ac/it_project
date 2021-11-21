from inference.engine import Model

indic2en_model = Model(expdir='../en-indic')

with open('/home3/181ee103/en_source_sentences.txt', 'r') as f:
    en_sent = f.readlines()

hi_translated = indic2en_model.batch_translate(en_sent, 'en', 'hi')

with open('/home3/181ee103/hi_translated_sentences.txt', 'w', encoding="utf8") as f:
	for s in hi_translated:
		if s[-1] == '\n':
			f.write(s)
		else:
			f.write(s + '\n')
	f.close()

kn_translated = indic2en_model.batch_translate(en_sent, 'en', 'kn')

with open('/home3/181ee103/kn_translated_sentences.txt', 'w', encoding="utf8") as f:
	for s in kn_translated:
		if s[-1] == '\n':
			f.write(s)
		else:
			f.write(s + '\n')
	f.close()


ta_translated = indic2en_model.batch_translate(en_sent, 'en', 'ta')

with open('/home3/181ee103/ta_translated_sentences.txt', 'w', encoding="utf8") as f:
	for s in ta_translated:
		if s[-1] == '\n':
			f.write(s)
		else:
			f.write(s + '\n')
	f.close()