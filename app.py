import streamlit as st  
import pandas as pd
import numpy as np
import os
import random
import time
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import poisson 

st.set_page_config(
    page_title = 'Previsão Esportiva - Copa do Mundo Qatar 2022',
    page_icon = '⚽',
    layout = 'centered',
    initial_sidebar_state = "expanded",
    menu_items = {
        'About': 'https://www.previsaoesportiva.com.br', 
    }
)


#@st.cache_data
def carrega_dados():
	url1 = 'https://docs.google.com/spreadsheets/d/1nkN0-CgrozB-s7rNTWtLNKEmhDpGNBO1Z1b7fMPq0aM/edit?usp=sharing'
	url2 = 'https://docs.google.com/spreadsheets/d/16EfqovJ-p3M9-P1zh1OXfcFNDuL3IsRE3NkUT4GqDx4/edit?usp=sharing'
	selecoes = pd.read_csv(url1.replace('/edit?usp=sharing', '/gviz/tq?tqx=out:csv'))
	selecoes.set_index('Seleção', inplace = True)
	tabela = pd.read_csv(url2.replace('/edit?usp=sharing', '/gviz/tq?tqx=out:csv'))
	return selecoes, tabela

selecoes, tabela = carrega_dados()



@st.cache_data
def Pontuacao(tabela):
    aux = pd.DataFrame(0, columns = ['Grupo','P', 'J', 'V', 'E', 'D', 'GP', 'GC', 'SG', '%'], index = selecoes.index)

    aux['Grupo'] = selecoes['Grupo']

    tabela_passado = tabela[tabela['Resultado'].apply(pd.notnull)]
    tabela_passado.reset_index(inplace = True)

    for linha in range(tabela_passado.shape[0]):
        s1, s2 = tabela_passado.loc[linha, ['Seleção1', 'Seleção2']]
        g1, g2 = tabela_passado.loc[linha, 'Resultado'].split('x')

        aux.loc[s1, 'GP'] += int(g1)
        aux.loc[s1, 'GC'] += int(g2)
        aux.loc[s2, 'GP'] += int(g2)
        aux.loc[s2, 'GC'] += int(g1)

        if g1 > g2:
            aux.loc[s1, 'V'] += 1
            aux.loc[s2, 'D'] += 1
        elif g1 == g2:
            aux.loc[s1, 'E'] += 1
            aux.loc[s2, 'E'] += 1
        else:
            aux.loc[s1, 'D'] += 1
            aux.loc[s2, 'V'] += 1

    aux['SG'] = aux['GP'] - aux['GC']
    aux['J'] = aux['V'] + aux['E'] + aux['D']
    aux['P'] = 3*aux['V'] + aux['E']
    aux['%'] = 100*aux['P']/(3*aux['J'])
    aux['%'] = aux['%'].apply(lambda x: f'{x:.0f}%')
    aux.sort_values(by = ['P', 'SG', 'GP'], ascending = False, inplace = True)

    return aux



@st.cache_data
def Força(tabela):

    def normalizar(vetor):
        min_val, max_val = min(vetor), max(vetor)
        if max_val - min_val == 0:
            saida = pd.Series(np.zeros(len(vetor)))
            saida.index = vetor.index
        else:
            saida = (vetor - min_val) / (max_val - min_val)
        return saida

    fifa = selecoes['RankingFIFA']
    fatorFifa = normalizar(fifa)

    saldo = Pontuacao(tabela)['SG']
    fatorSaldo = normalizar(saldo)

    copas = selecoes['Copas']
    fatorCopas = normalizar(copas)

    forca = normalizar(fatorFifa**3 + 0.1*fatorSaldo + 0.05*fatorCopas) + 0.2

    aux = pd.DataFrame({'Fifa': fatorFifa**3,
                'Saldo': 0.1*fatorSaldo,
                'Copas': 0.05*fatorCopas,
                'Força': forca }).sort_values(by = 'Força', ascending = False)

    return forca.sort_values(ascending = False) , aux

forca = Força(tabela)[0]




def SimulaJogoPrimeiraFase(sele1, sele2):

    def Forca(sele1, sele2):
        forca1, forca2 = forca[sele1], forca[sele2]
        return forca1, forca2

    def MediasPoisson(sele1, sele2):
        forca1, forca2 = Forca(sele1, sele2)
        fator = forca1/(forca1 + forca2)
        mgols = MEDIA_GOLS_COPA    #MEDIA DE GOLS ESPERADA NA COPA - PARAMETRO IMPORTANTE!
        media1 = mgols*fator
        media2 = mgols - media1
        return media1, media2

    def Distribuicao(media, tamanho = 7):
        probs = [poisson.pmf(i, media) for i in range(tamanho)]
        probs.append(1-sum(probs))
        return pd.Series(probs, index = ['0', '1', '2', '3', '4', '5', '6', '7+'])

    def ProbabilidadesPartida(sele1, sele2):
        l1, l2 = MediasPoisson(sele1, sele2)
        d1, d2 = Distribuicao(l1), Distribuicao(l2)
        matriz = np.outer(d1, d2)    #   Monta a matriz de probabilidades

        vitoria = np.tril(matriz).sum()-np.trace(matriz)    #Soma a triangulo inferior
        derrota = np.triu(matriz).sum()-np.trace(matriz)    #Soma a triangulo superior
        probs = np.around([vitoria, 1-(vitoria+derrota), derrota], 3)
        probsp = [f'{100*i:.1f}%' for i in probs]

        nomes = ['0', '1', '2', '3', '4', '5', '6', '7+']
        matriz = pd.DataFrame(matriz, columns = nomes, index = nomes)
        matriz.index = pd.MultiIndex.from_product([[sele1], matriz.index])
        matriz.columns = pd.MultiIndex.from_product([[sele2], matriz.columns])

        return probsp, matriz

    l1, l2 = MediasPoisson(sele1, sele2)
    gols1 = int(np.random.poisson(lam = l1, size = 1))
    gols2 = int(np.random.poisson(lam = l2, size = 1))
    placar = f'{gols1}x{gols2}'
    probs, matriz = ProbabilidadesPartida(sele1, sele2)

    return sele1, sele2, placar, probs, matriz



def SimulaJogoMataMata(selecao1, selecao2):
    placar, probs = SimulaJogoPrimeiraFase(selecao1, selecao2)[2:4]
    g1, g2 = placar.split('x')
    vencedor = selecao1 if g1 > g2 else selecao2 if g1 < g2 else random.sample([selecao1, selecao2], 1)[0]
    p = float(probs[0][:-1]) + float(probs[1][:-1])/2
    probs = [f'{i:.1f}%' for i in [p, 100-p]]
    return vencedor, selecao1, selecao2, placar, probs





def SimulaCopa(tabela):

    tabela_sim = tabela.copy()

    ## TABELA COM OS ACONTECIMENTOS DA COPA
    cols = ['1st', '2nd', '3th', '4th', 'Oitavas', 'Quartas', 'Semis', 'Final', 'Campeão']
    info = pd.DataFrame(0, columns = cols, index = selecoes.index)


    ## COMPLETA TODOS OS JOGOS FALTANTES DA PRIMEIRA FASE
    for i in range(48):
        if pd.isnull(tabela_sim.loc[i, 'Resultado']):
            s1 = tabela_sim.loc[i, 'Seleção1']
            s2 = tabela_sim.loc[i, 'Seleção2']
            tabela_sim.loc[i, 'Resultado'] = SimulaJogoPrimeiraFase(s1, s2)[2]

    ## SUMARIZA OS RESULTADOS DA PRIMEIRA FASE
    pontuacao = Pontuacao(tabela_sim)

    ## SEPARA AS SELECOES QUE AVANÇARAM PARA AS OITAVAS - TOP16
    top16 = {}
    for grupo in list('ABCDEFGH'):
        resultado_grupo = pontuacao[pontuacao['Grupo'] == grupo]
        top16[f'1{grupo}'] = resultado_grupo.index[0]   #guarda no dicionario o 1º do grupo com a indentificacao adequada
        top16[f'2{grupo}'] = resultado_grupo.index[1]   #guarda no dicionario o 2º do grupo com a indentificacao adequada
        for i, j in zip(resultado_grupo.index, cols[:4]):
            info.loc[i,j] += 1

    ## PREENCHE NA TABELA AS SELECOES QUE AVANCARAM PARA OITAVAS
    tabela_sim.replace(top16, inplace = True)


    def SimulaEtapa(tabela, etapa): #OITAVAS, QUARTAS, SEMIS
        indices = tabela[tabela['Rodada'] == etapa].index.tolist()   #indices na tabela
        avanca = {}
        for i in indices:
            if pd.isnull(tabela.loc[i, 'Resultado']): # preenche apenas o que nao está preenchido
                jogo = SimulaJogoMataMata(tabela.loc[i, 'Seleção1'], tabela.loc[i, 'Seleção2'])
                avanca[tabela.loc[i, 'Grupo']] = jogo[0]  #guarda no dicionario quem venceu
                tabela.loc[i, 'Resultado'] = jogo[3] #preenche a tabela com o placar simulado
        tabela['Seleção1'].replace(avanca, inplace = True)
        tabela['Seleção2'].replace(avanca, inplace = True)
        return avanca, tabela

    # SIMULAÇÃO DAS ETAPAS DA SEGUNDA FASE
    top8, tabela_sim = SimulaEtapa(tabela_sim, 'OITAVAS')
    top4, tabela_sim = SimulaEtapa(tabela_sim, 'QUARTAS')
    top2, tabela_sim = SimulaEtapa(tabela_sim, 'SEMIS')

    # TERCEIRO E QUARTO FINALISTAS
    teq = list(set(top4.values()) - set(top2.values()))
    tabela_sim.replace({'PS1': teq[0], 'PS2': teq[1]}, inplace = True)
    terceiro = SimulaJogoMataMata(top2['S1'], top2['S2'])
    tabela_sim.loc[62, 'Resultado'] = terceiro[3]

    # GRANDE FINAL
    final = SimulaJogoMataMata(top2['S1'], top2['S2'])
    tabela_sim.loc[63, 'Resultado'] = final[3]
    top1 = {'Campeão': final[0]}

    ## ATUALIZACAO DA MATRIZ DE INFORMAÇÕES
    info.loc[list(top16.values()), 'Oitavas'] += 1
    info.loc[list(top8.values()), 'Quartas'] += 1
    info.loc[list(top4.values()), 'Semis'] += 1
    info.loc[list(top2.values()), 'Final'] += 1
    info.loc[list(top1.values()), 'Campeão'] += 1

    return info, list(top2.values()), tabela_sim, Pontuacao(tabela_sim)



### PARAMETROS UTILIZADOS
MEDIA_GOLS_COPA = 3 


def Texto(texto, tamanho, cor, alinhamento, tipo):
	st.markdown(f"<{tipo} style='text-align: {alinhamento}; color: {cor}; font-size: {tamanho}px;'> {texto} </{tipo}>", unsafe_allow_html=True)

###### CABEÇALHO #######
_, col, _ = st.columns([1,1,1])

with col:
	st.image('imagens/previsaoesportivalogo.png', use_column_width = True)

Texto(texto = 'Previsão Esportiva - Copa do Mundo Feminina 2023 🏆', 
	  tamanho = 25, 
	  cor = '#0f54c9', 
	  alinhamento = 'center', 
	  tipo = 'h1')
	
st.image('imagens/banner-copa-feminina-m.png', use_column_width = True)
 
st.markdown('---')



###### ESCOLHA DA PÁGINA #######

if 'pagina' not in st.session_state:
    st.session_state['pagina'] = 'pagina1'

col1, col2, col3 = st.columns([1,1,1])

with col1: 
	if st.button('⚽ PREVISÃO DE JOGOS', type="primary", use_container_width=True):
		st.session_state['pagina'] = 'pagina1'

with col2:
	if st.button('🏆 PREVISÃO DA COPA', type="primary", use_container_width=True):
		st.session_state['pagina'] = 'pagina2'

with col3:
	if st.button('📊 TODOS OS NÚMEROS', type="primary", use_container_width=True):
		st.session_state['pagina'] = 'pagina3'

st.markdown('---')


###### PAGINA 1 #######
if st.session_state['pagina'] == 'pagina1':
	Texto(texto = '⚽ Previsão de Jogos', tamanho = 32, cor = '#0f54c9', alinhamento = 'center', tipo = 'h3')

	
 

	listaselecoes = selecoes.index.tolist()  
	listaselecoes.sort()
	listaselecoes2 = listaselecoes.copy()


	c1, c2 = st.columns([1,2])
	with c1:
		#Texto('Escolha o tipo de jogo:')

		Texto(texto = 'Escolha o tipo de jogo:', tamanho = 16, cor = '#0f0f0f', alinhamento = 'right', tipo = 'p')

	with c2:
		tipojogo = st.radio('Escolha o tipo de jogo', ['Jogo da Fase de Grupos', 'Jogo do Mata-Mata'],
							 label_visibility = 'collapsed',
			 				 horizontal = True)
	st.markdown('---')

	j1, j2 = st.columns (2)
	selecao1 = j1.selectbox('--- Escolha a primeira Seleção ---', listaselecoes, index = 3) 

	listaselecoes2.remove(selecao1)
	selecao2 = j2.selectbox('--- Escolha a segunda Seleção ---', listaselecoes2, index = 2)
	 
	st.markdown('---')
 


	jogo = SimulaJogoPrimeiraFase(selecao1, selecao2)
	prob = jogo[3]
	matriz = jogo[4]

 
	if tipojogo == 'Jogo da Fase de Grupos':
		col1, col2, col3, col4, col5 = st.columns(5)
		col1.image(selecoes.loc[selecao1, 'LinkBandeira']) 
		col2.markdown(f"<h5 style='text-align: center; color: #1a1a1a; font-weight: bold; font-size: 25px;'>{selecao1}<br>  </h1>", unsafe_allow_html=True)
		col2.markdown(f"<h2 style='text-align: center; color: #0f54c9; font-weight: bold; font-size: 50px;'>{prob[0]}<br>  </h1>", unsafe_allow_html=True)
		col3.markdown(f"<h2 style='text-align: center; color: #6a6a6b; font-weight: 100; font-size: 15px;'>Empate<br>  </h1>", unsafe_allow_html=True)
		col3.markdown(f"<h2 style='text-align: center; color: #6a6a6b;                    font-size: 30px;'>{prob[1]}<br>  </h1>", unsafe_allow_html=True)
		col4.markdown(f"<h5 style='text-align: center; color: #1a1a1a; font-weight: bold; font-size: 25px;'>{selecao2}<br>  </h1>", unsafe_allow_html=True) 
		col4.markdown(f"<h2 style='text-align: center; color: #0f54c9; font-weight: bold; font-size: 50px;'>{prob[2]}<br>  </h1>", unsafe_allow_html=True) 
		col5.image(selecoes.loc[selecao2, 'LinkBandeira'])
 
	if tipojogo == 'Jogo do Mata-Mata':
		col1, col2, col3, col4, col5 = st.columns(5)
		col1.image(selecoes.loc[selecao1, 'LinkBandeira'])  
		col2.markdown(f"<h5 style='text-align: center; color: #1a1a1a; font-weight: bold; font-size: 25px;'>{selecao1}<br>  </h1>", unsafe_allow_html=True)
		aux1 = round(float(prob[0][:-1])+float(prob[1][:-1])/2, 1)
		aux2 = str(aux1) + '%' 
		col2.markdown(f"<h2 style='text-align: center; color: #0f54c9; font-weight: bold; font-size: 50px;'>{aux2}<br>  </h1>", unsafe_allow_html=True)
		col3.markdown(f"<h2 style='text-align: center; color: #6a6a6b; font-weight: 100; font-size: 15px;'> <br>  </h1>", unsafe_allow_html=True)
		col3.markdown(f"<h2 style='text-align: center; color: #6a6a6b;                    font-size: 30px;'>vs<br>  </h1>", unsafe_allow_html=True)
		col4.markdown(f"<h5 style='text-align: center; color: #1a1a1a; font-weight: bold; font-size: 25px;'>{selecao2}<br>  </h1>", unsafe_allow_html=True) 
		aux3 = round(100 - aux1, 1)
		aux4 = str(aux3) + '%' 
		col4.markdown(f"<h2 style='text-align: center; color: #0f54c9; font-weight: bold; font-size: 50px;'>{aux4}<br>  </h1>", unsafe_allow_html=True) 
		col5.image(selecoes.loc[selecao2, 'LinkBandeira'])

	st.markdown('---')
 
 

	lista07 = ['0', '1', '2', '3', '4', '5', '6', '7+']
	
	fig, ax = plt.subplots()
	sns.heatmap(matriz.reset_index(drop=True), ax=ax, cmap = 'Blues', annot = matriz , fmt=".1%", xticklabels = lista07, yticklabels = lista07) 
	ax.tick_params(axis='both', which='major', labelsize=8, labelbottom = False, bottom=False, top = True, labeltop=True )
	ax.xaxis.set_label_position('top')
	ax.set_xlabel('Gols ' + selecao2, fontsize=12, color = '#363636')	
	ax.set_ylabel('Gols ' + selecao1, fontsize=12, color = '#363636')	
	ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 8, color = '#363636')
	ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 8, color = '#363636' )

	Texto(texto = '<br>⚽ Probabilidades dos Placares da Partida<br><br>', tamanho = 32, cor = '#0f54c9', alinhamento = 'center', tipo = 'h3')

	#st.markdown("<h2 style='text-align: center; color: #0f54c9; font-size: 40px;'> Probabilidades dos Placares ⚽<br>  </h1>", unsafe_allow_html=True) 
	st.write(fig) 

	placar = np.unravel_index(np.argmax(matriz, axis=None), matriz.shape) 

 

	_, c1, c2, _ = st.columns([1,4,4,1])
	with c1:
		Texto('Placar Mais Provável:', 25, cor = '#0f0f0f', alinhamento = 'right', tipo = 'p'  )

	with c2:
		col1, col2, col3, _ = st.columns([1.5, 1.5, 1.5, 2.5])
		with col1:
			st.image(selecoes.loc[selecao1, 'LinkBandeira'])
		with col2:
			Texto(f'<b>{placar[0]}x{placar[1]}</b>', 30, cor = '#0f0f0f', alinhamento = 'center', tipo = 'p'  )		
		with col3:
			st.image(selecoes.loc[selecao2, 'LinkBandeira'])





###### PAGINA 2 #######
if st.session_state['pagina'] == 'pagina2':
	Texto(texto = '🏆 Previsão da Copa', tamanho = 32, cor = '#0f54c9', alinhamento = 'center', tipo = 'h3')
	
	Texto(texto = '<br><b>Jogos que já ocorreram ⬇</b>', tamanho = 20, cor = 'black', alinhamento = 'center', tipo = 'p')
	indice_ate_momento = tabela['Resultado'].last_valid_index()+1

	passado = tabela.iloc[:tabela['Resultado'].last_valid_index()+1] 		

	colunas = ['Data', 'Hora', 'Rodada', 'Grupo', 'Seleção1', 'Resultado', 'Seleção2']

	st.dataframe(passado[colunas],
		 use_container_width=True,
		 hide_index=True,
		 height = int(passado.shape[0]*36.1) )
	
	#Texto(texto = '<br><b>Jogos que ainda vão acontecer... SIMULADOS! 🔮</b>', tamanho = 20, cor = 'black', alinhamento = 'center', tipo = 'p')
	
	botao = st.button('🔮 Clique para Simular o Restante da Copa!', type="primary", use_container_width=True)

	if botao:
		sim = SimulaCopa(tabela)
		futuro = sim[2].iloc[indice_ate_momento:] 
	 
		st.dataframe(futuro[colunas],
			 use_container_width=True,
			 hide_index=True,
			 height =  int(futuro.shape[0]*36.3) )

		vencedor = sim[0].sort_values(by = 'Campeão', ascending = False).index[0]

		Texto(texto = f'Nessa simulação, a seleção <b>{vencedor}</b> vence a Copa do Mundo 2023', 
			tamanho = 20, cor = '#161616', alinhamento = 'center', tipo = 'p')
		




###### PAGINA 3 #######
if st.session_state['pagina'] == 'pagina3':
	Texto(texto = '📊 Todos os Números', tamanho = 32, cor = '#0f54c9', alinhamento = 'center', tipo = 'h3')
	arquivos = sorted(list(set([i[:10] for i in os.listdir('outputs')])))
	ultimo = arquivos.pop()
	arquivos.insert(0, ultimo)
	def formatacao(x):
		if x == 'InicioCopa':
			return 'Início da Competição'
		else:
			return f'Resultados até dia {x[:2]}/{x[3:5]}/{x[-4:]}'
	at = st.selectbox('Atualização', arquivos, format_func = formatacao)
	 
	abas = ['Probabilidades de Avançar', 'Probabilidade de Cair', 'Finais Mais Prováveis']
	abas = st.tabs(abas)

	with abas[0]: 
		st.dataframe(pd.read_excel(f'outputs/{at}-SimulaçõesCopa.xlsx'),
		 use_container_width=True,
		 hide_index=True,
		 height = 1170 )

	with abas[1]:
		st.dataframe(pd.read_excel(f'outputs/{at}-ProbabilidadeCair.xlsx'),
		 use_container_width=True,
		 hide_index=True,
		 height = 1170 )

	with abas[2]:
		st.dataframe(pd.read_excel(f'outputs/{at}-FinaisMaisProvaveis.xlsx'),
		 use_container_width=True,
		 hide_index=True,
		 height = 1170 )


####### RODAPÉ ####### 
st.markdown('---')
st.markdown('Trabalho desenvolvido pela Equipe Previsão Esportiva - acesse www.previsaoesportiva.com.br 🔗')
 
# st.markdown("<h2 style='text-align: center; color: #0f54c9; font-size: 40px;'>Probabilidades dos Jogos ⚽<br>  </h1>", unsafe_allow_html=True)



# dados_variaveis = pd.read_excel('dados_previsao_esportiva.xlsx', sheet_name ='grupos')
# fifa = dados_variaveis['Ranking Point']
# fifa.index = dados_variaveis['Seleção']

# a, b = min(fifa), max(fifa) 
# fa, fb = 0.05, 1 
# b1 = (fb - fa)/(b-a) 
# b0 = fb - b*b1
# fatorFifa = b0 + b1*fifa 

# fatorFifa.sort_values(ascending = False)

# fifa = dados_variaveis['RankingELO']
# fifa.index = dados_variaveis['Seleção']

# a, b = min(fifa), max(fifa) 
# fa, fb = 0.05, 1 
# b1 = (fb - fa)/(b-a) 
# b0 = fb - b*b1
# fatorELO = b0 + b1*fifa 

# fatorELO.sort_values(ascending = False)

# def Fator(dados, var, K):
#     res = K * (dados[var] - min(dados[var]))/(max(dados[var]) - min(dados[var])) + (1 - K)
#     res.index = dados_variaveis['Seleção']
#     return res

# fatorMercado = Fator(dados_variaveis, 'Market Value', K = 0.1) 
# fatorATQ = Fator(dados_variaveis, 'ATAQUE', K = 0.05) 
# fatorDEF = 1 - Fator(dados_variaveis, 'DEFESA', K = 0.05) + 0.95
# fatorCopa = Fator(dados_variaveis, 'Copas2', K = 0.1)
# fatorTendencia = Fator(dados_variaveis, 'Saldo', K = 0.1)

# fatores =  (fatorMercado * fatorDEF * fatorATQ * fatorCopa * fatorTendencia)

# forca = (0.5*fatorFifa + 0.5*fatorELO) * fatores
# forca = forca/max(forca)
# forca = 0.7*(forca - min(forca))/(max(forca) - min(forca)) + 0.30
# forca = forca.sort_values(ascending = False) 

# lista07 = ['0', '1', '2', '3', '4', '5', '6', '7+']

# def Resultado(gols1, gols2):
# 	if gols1 > gols2:
# 		res = 'V'
# 	if gols1 < gols2:
# 		res = 'D' 
# 	if gols1 == gols2:
# 		res = 'E'       
# 	return res

# def MediasPoisson(sele1, sele2):
# 	forca1 = forca[sele1]
# 	forca2 = forca[sele2]
# 	fator = forca1/(forca1 + forca2)
# 	mgols = 2.5
# 	l1 = mgols*fator
# 	l2 = mgols - l1
# 	return [fator, l1, l2]
	
# def Distribuicao(media, tamanho = 7):
# 	probs = []
# 	for i in range(tamanho):
# 		probs.append(poisson.pmf(i,media))
# 	probs.append(1-sum(probs))
# 	return pd.Series(probs, index = lista07)

# def ProbabilidadesPartida(sele1, sele2):
# 	fator, l1, l2 = MediasPoisson(sele1, sele2)
# 	d1, d2 = Distribuicao(l1), Distribuicao(l2)  
# 	matriz = np.outer(d1, d2)    #   Monta a matriz de probabilidades

# 	vitoria = np.tril(matriz).sum()-np.trace(matriz)    #Soma a triangulo inferior
# 	derrota = np.triu(matriz).sum()-np.trace(matriz)    #Soma a triangulo superior
# 	probs = np.around([vitoria, 1-(vitoria+derrota), derrota], 3)
# 	probsp = [f'{100*i:.1f}%' for i in probs]

# 	nomes = ['0', '1', '2', '3', '4', '5', '6', '7+']
# 	matriz = pd.DataFrame(matriz, columns = nomes, index = nomes)
# 	matriz.index = pd.MultiIndex.from_product([[sele1], matriz.index])
# 	matriz.columns = pd.MultiIndex.from_product([[sele2], matriz.columns]) 
# 	output = {'seleção1': sele1, 'seleção2': sele2, 
# 			 'f1': forca[sele1], 'f2': forca[sele2], 'fator': fator, 
# 			 'media1': l1, 'media2': l2, 
# 			 'probabilidades': probsp, 'matriz': matriz}
# 	return output

# def Pontos(gols1, gols2):
# 	rst = Resultado(gols1, gols2)
# 	if rst == 'V':
# 		pontos1, pontos2 = 3, 0
# 	if rst == 'E':
# 		pontos1, pontos2 = 1, 1
# 	if rst == 'D':
# 		pontos1, pontos2 = 0, 3
# 	return pontos1, pontos2, rst


# def Jogo(sele1, sele2):
# 	fator, l1, l2 = MediasPoisson(sele1, sele2)
# 	gols1 = int(np.random.poisson(lam = l1, size = 1))
# 	gols2 = int(np.random.poisson(lam = l2, size = 1))
# 	saldo1 = gols1 - gols2
# 	saldo2 = -saldo1
# 	pontos1, pontos2, result = Pontos(gols1, gols2)
# 	placar = '{}x{}'.format(gols1, gols2)
# 	return [gols1, gols2, saldo1, saldo2, pontos1, pontos2, result, placar]


# listaselecoes = dados_variaveis['Seleção'].tolist()  
# listaselecoes.sort()
# listaselecoes2 = listaselecoes.copy()

# ######## COMEÇO DO APP

# paginas = ['Principal', 'Tabelas']
# pagina = st.sidebar.radio('Selecione a página', paginas)



# if pagina == 'Principal':

# 	a1, a2 = st.columns([1,4])
# 	a1.image('imagens/previsaoesportivalogo.png', width = 200)
# 	a2.markdown("<h2 style='text-align: right; color: #5C061E; font-size: 32px;'>Copa do Mundo Qatar 2022 🏆  </h1>", unsafe_allow_html=True)
# 	st.markdown('---')
# 	st.markdown("<h2 style='text-align: center; color: #0f54c9; font-size: 40px;'>Probabilidades dos Jogos ⚽<br>  </h1>", unsafe_allow_html=True)

# 	st.markdown('---')
# 	tipojogo = st.radio('Escolha o tipo de jogo', ['Jogo da Fase de Grupos', 'Jogo do Mata-Mata'])
# 	st.markdown('---')

# 	j1, j2 = st.columns (2)
# 	selecao1 = j1.selectbox('--- Escolha a primeira Seleção ---', listaselecoes) 
# 	listaselecoes2.remove(selecao1)
# 	selecao2 = j2.selectbox('--- Escolha a segunda Seleção ---', listaselecoes2, index = 1)
	 
# 	st.markdown('---')
 


# 	jogo = ProbabilidadesPartida(selecao1, selecao2)
# 	prob = jogo['probabilidades']
# 	matriz = jogo['matriz']

 
# 	if tipojogo == 'Jogo da Fase de Grupos':
# 		col1, col2, col3, col4, col5 = st.columns(5)
# 		col1.image(dados_variaveis[dados_variaveis['Seleção'] == selecao1]['LinkBandeira2'].iloc[0]) 
# 		col2.markdown(f"<h5 style='text-align: center; color: #1a1a1a; font-weight: bold; font-size: 25px;'>{selecao1}<br>  </h1>", unsafe_allow_html=True)
# 		col2.markdown(f"<h2 style='text-align: center; color: #0f54c9; font-weight: bold; font-size: 50px;'>{prob[0]}<br>  </h1>", unsafe_allow_html=True)
# 		col3.markdown(f"<h2 style='text-align: center; color: #6a6a6b; font-weight: 100; font-size: 15px;'>Empate<br>  </h1>", unsafe_allow_html=True)
# 		col3.markdown(f"<h2 style='text-align: center; color: #6a6a6b;                    font-size: 30px;'>{prob[1]}<br>  </h1>", unsafe_allow_html=True)
# 		col4.markdown(f"<h5 style='text-align: center; color: #1a1a1a; font-weight: bold; font-size: 25px;'>{selecao2}<br>  </h1>", unsafe_allow_html=True) 
# 		col4.markdown(f"<h2 style='text-align: center; color: #0f54c9; font-weight: bold; font-size: 50px;'>{prob[2]}<br>  </h1>", unsafe_allow_html=True) 
# 		col5.image(dados_variaveis[dados_variaveis['Seleção'] == selecao2]['LinkBandeira2'].iloc[0])
 
# 	if tipojogo == 'Jogo do Mata-Mata':
# 		col1, col2, col3, col4, col5 = st.columns(5)
# 		col1.image(dados_variaveis[dados_variaveis['Seleção'] == selecao1]['LinkBandeira2'].iloc[0]) 
# 		col2.markdown(f"<h5 style='text-align: center; color: #1a1a1a; font-weight: bold; font-size: 25px;'>{selecao1}<br>  </h1>", unsafe_allow_html=True)
# 		aux1 = round(float(prob[0][:-1])+float(prob[1][:-1])/2, 1)
# 		aux2 = str(aux1) + '%' 
# 		col2.markdown(f"<h2 style='text-align: center; color: #0f54c9; font-weight: bold; font-size: 50px;'>{aux2}<br>  </h1>", unsafe_allow_html=True)
# 		col3.markdown(f"<h2 style='text-align: center; color: #6a6a6b; font-weight: 100; font-size: 15px;'> <br>  </h1>", unsafe_allow_html=True)
# 		col3.markdown(f"<h2 style='text-align: center; color: #6a6a6b;                    font-size: 30px;'>vs<br>  </h1>", unsafe_allow_html=True)
# 		col4.markdown(f"<h5 style='text-align: center; color: #1a1a1a; font-weight: bold; font-size: 25px;'>{selecao2}<br>  </h1>", unsafe_allow_html=True) 
# 		aux3 = round(100 - aux1, 1)
# 		aux4 = str(aux3) + '%' 
# 		col4.markdown(f"<h2 style='text-align: center; color: #0f54c9; font-weight: bold; font-size: 50px;'>{aux4}<br>  </h1>", unsafe_allow_html=True) 
# 		col5.image(dados_variaveis[dados_variaveis['Seleção'] == selecao2]['LinkBandeira2'].iloc[0])

# 	st.markdown('---')
 


# 	def aux(x):
# 		return f'{str(round(100*x,2))}%'

# 	#st.table(matriz.applymap(aux))
 


	
# 	fig, ax = plt.subplots()
# 	sns.heatmap(matriz.reset_index(drop=True), ax=ax, cmap = 'Blues', annot = 100*matriz , fmt=".2f", xticklabels = lista07, yticklabels = lista07) 
# 	ax.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = True, labeltop=True )
# 	ax.xaxis.set_label_position('top')
# 	ax.set_xlabel('Gols ' + selecao2, fontsize=15, color = 'gray')	
# 	ax.set_ylabel('Gols ' + selecao1, fontsize=15, color = 'gray')	
# 	ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 8, color = 'gray')
# 	ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 8, color = 'gray' )


# 	st.markdown("<h2 style='text-align: center; color: #0f54c9; font-size: 40px;'> Probabilidades dos Placares ⚽<br>  </h1>", unsafe_allow_html=True) 
# 	st.write(fig)

# 	st.markdown('---')

# 	placar = np.unravel_index(np.argmax(matriz, axis=None), matriz.shape) 

# 	st.markdown("<h2 style='text-align: center; color: #0f54c9; font-size: 40px;'> Placar Mais Provável ⚽<br>  </h1>", unsafe_allow_html=True)
	
# 	st.markdown(' ')

# 	col1, col2, col3 = st.columns([1,5,1])
# 	col1.image(dados_variaveis[dados_variaveis['Seleção'] == selecao1]['LinkBandeira2'].iloc[0]) 
# 	#col2.header(selecao1) 
# 	col2.markdown(f"<h2 style='text-align: center; color: #1a1a1a; font-size: 40px;'>{selecao1} {placar[0]}x{placar[1]} {selecao2}<br>  </h1>", unsafe_allow_html=True)
# 	#col4.header(selecao2)
# 	col3.image(dados_variaveis[dados_variaveis['Seleção'] == selecao2]['LinkBandeira2'].iloc[0]) 



# 	st.markdown('---')

# 	st.markdown('Trabalho desenvolvido pela Equipe Previsão Esportiva - acesse www.previsaoesportiva.com.br 🔗')

# 	#bandeira1, nome1, prob, empate, prob, nome2, bandeira2
# 	#matriz de probabilidades do jogo
# 	#placar mais provável

# if pagina == 'Tabelas': 

# 	atualizacoes = ['Início da Copa', 'Pós Primeira Rodada', 'Pós Segunda Rodada', 'Oitavas de Final','Quartas de Final', 'Semifinais', 'Final']
# 	a = st.radio('Selecione a Atualização', atualizacoes, index = 5)

# 	if a == 'Início da Copa':
# 		dados0 = pd.read_excel('dados_previsao_esportiva.xlsx', sheet_name ='grupos', index_col=0) 
# 		dados1 = pd.read_excel('dados/outputSimulaçõesCopa(n=1000000).xlsx', index_col=0) 
# 		dados2 = pd.read_excel('dados/outputJogadoresArtilharia(n=1000000).xlsx', index_col=0) 
# 		dados3 = pd.read_excel('dados/outputFinaisMaisProvaveis(n=1000000).xlsx', index_col=0) 
# 		dados4 = pd.read_excel('dados/outputProbPorEtapa(n=1000000).xlsx', index_col=0) 
# 		dados5 = pd.read_excel('dados/outputTabelaJogosPROBS.xlsx', index_col=0) 

# 		tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(['Dados das Seleções', "Simulações da Copa", "Artilheiro", "Finais Mais Prováveis",  'Probabilidades por Etapa', 'Tabela de Jogos'])

# 		with tab0:
# 			st.header("Dados das Seleções") 
# 			st.write(dados0)

# 		with tab1:
# 			st.header("Simulações da Copa") 
# 			st.write(dados1)

# 		with tab2:  
# 			st.header("Previsões do Artilheiro")  
# 			st.write(dados2)

# 		with tab3:  
# 			st.header("Finais Mais Prováveis")  
# 			st.write(dados3) 

# 		with tab4:  
# 			st.header("Probabilidades por Etapa")  
# 			st.write(dados4) 

# 		with tab5:  
# 			st.header("Tabela de Jogos")  
# 			st.write(dados5[['grupo', 'seleção1', 'probV', 'probE', 'probD','seleção2']])  

# 	if a == 'Pós Primeira Rodada':
# 		dados1 = pd.read_excel('dados/R1outputSimulaçõesCopa(n=1000000).xlsx', index_col=0) 
# 		dados2 = pd.read_excel('dados/R1outputJogadoresArtilharia(n=1000000).xlsx', index_col=0) 
# 		dados3 = pd.read_excel('dados/R1outputFinaisMaisProvaveis(n=1000000).xlsx', index_col=0) 
# 		dados4 = pd.read_excel('dados/R1outputProbPorEtapa(n=1000000).xlsx', index_col=0) 
# 		dados5 = pd.read_excel('dados/R1outputTabelaJogosPROBS.xlsx', index_col=0) 
# 		dados6 = pd.read_excel('dados/R1outputAvançoPorEtapa.xlsx', index_col=0) 

# 		tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Simulações da Copa", 'Artilharia', "Finais Mais Prováveis",  'Probabilidades por Etapa', 'Tabela de Jogos','Probabilidades de Avanço'])
 
# 		with tab1:
# 			st.header("Simulações da Copa") 
# 			st.write(dados1)

# 		with tab2:  
# 			st.header("Previsões do Artilheiro")  
# 			st.write(dados2)

# 		with tab3:  
# 			st.header("Finais Mais Prováveis")  
# 			st.write(dados3) 

# 		with tab4:  
# 			st.header("Probabilidades por Etapa")  
# 			st.write(dados4) 

# 		with tab5:  
# 			st.header("Tabela de Jogos")  
# 			st.write(dados5[['grupo', 'seleção1', 'probV', 'probE', 'probD','seleção2']])  

# 		with tab6:  
# 			st.header("Probabilidades de Avanço")  
# 			st.write(dados6) 



# 	if a == 'Pós Segunda Rodada':
# 		dados1 = pd.read_excel('dados/R2outputSimulaçõesCopa(n=1000000).xlsx', index_col=0) 
# 		dados2 = pd.read_excel('dados/R2outputJogadoresArtilharia(n=1000000).xlsx', index_col=0) 
# 		dados3 = pd.read_excel('dados/R2outputFinaisMaisProvaveis(n=1000000).xlsx', index_col=0) 
# 		dados4 = pd.read_excel('dados/R2outputProbPorEtapa(n=1000000).xlsx', index_col=0) 
# 		dados5 = pd.read_excel('dados/R2outputTabelaJogosPROBS.xlsx', index_col=0) 
# 		dados6 = pd.read_excel('dados/R2outputAvançoPorEtapa.xlsx', index_col=0) 

# 		tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Simulações da Copa", 'Artilharia', "Finais Mais Prováveis",  'Probabilidades por Etapa', 'Tabela de Jogos','Probabilidades de Avanço'])
 
# 		with tab1:
# 			st.header("Simulações da Copa") 
# 			st.write(dados1)

# 		with tab2:  
# 			st.header("Previsões do Artilheiro")  
# 			st.write(dados2)

# 		with tab3:  
# 			st.header("Finais Mais Prováveis")  
# 			st.write(dados3) 

# 		with tab4:  
# 			st.header("Probabilidades por Etapa")  
# 			st.write(dados4) 

# 		with tab5:  
# 			st.header("Tabela de Jogos")  
# 			st.write(dados5[['grupo', 'seleção1', 'probV', 'probE', 'probD','seleção2']])  

# 		with tab6:  
# 			st.header("Probabilidades de Avanço")  
# 			st.write(dados6) 



# 	if a == 'Oitavas de Final':
# 		dados1 = pd.read_excel('dados/R3outputSimulaçõesCopa(n=1000000).xlsx', index_col=0) 
# 		dados2 = pd.read_excel('dados/R3outputJogadoresArtilharia(n=1000000).xlsx', index_col=0) 
# 		dados3 = pd.read_excel('dados/R3outputFinaisMaisProvaveis(n=1000000).xlsx', index_col=0) 
# 		dados4 = pd.read_excel('dados/R3outputProbPorEtapa(n=1000000).xlsx', index_col=0)  
# 		dados6 = pd.read_excel('dados/R3outputAvançoPorEtapa.xlsx', index_col=0) 

# 		tab1, tab2, tab3, tab4, tab5 = st.tabs(["Simulações da Copa", 'Artilharia', "Finais Mais Prováveis",  'Probabilidades por Etapa', 'Probabilidades de Avanço'])
 
# 		with tab1:
# 			st.header("Simulações da Copa") 
# 			st.write(dados1)

# 		with tab2:  
# 			st.header("Previsões do Artilheiro")  
# 			st.write(dados2)

# 		with tab3:  
# 			st.header("Finais Mais Prováveis")  
# 			st.write(dados3) 

# 		with tab4:  
# 			st.header("Probabilidades por Etapa")  
# 			st.write(dados4)  

# 		with tab5:  
# 			st.header("Probabilidades de Avanço")  
# 			st.write(dados6) 

# if a == 'Quartas de Final':
# 		dados1 = pd.read_excel('dados/R4outputSimulaçõesCopa(n=1000000).xlsx', index_col=0) 
# 		dados2 = pd.read_excel('dados/R4outputJogadoresArtilharia(n=1000000).xlsx', index_col=0) 
# 		dados3 = pd.read_excel('dados/R4outputFinaisMaisProvaveis(n=1000000).xlsx', index_col=0) 
# 		dados4 = pd.read_excel('dados/R4outputProbPorEtapa(n=1000000).xlsx', index_col=0)  
# 		dados6 = pd.read_excel('dados/R4outputAvançoPorEtapa.xlsx', index_col=0) 

# 		tab1, tab2, tab3, tab4, tab5 = st.tabs(["Simulações da Copa", 'Artilharia', "Finais Mais Prováveis",  'Probabilidades por Etapa','Probabilidades de Avanço'])
 
# 		with tab1:
# 			st.header("Simulações da Copa") 
# 			st.write(dados1)

# 		with tab2:  
# 			st.header("Previsões do Artilheiro")  
# 			st.write(dados2) 

# 		with tab3:  
# 			st.header("Finais Mais Prováveis")  
# 			st.write(dados3) 

# 		with tab4:  
# 			st.header("Probabilidades por Etapa")  
# 			st.write(dados4) 

# 		with tab5:  
# 			st.header("Probabilidades de Avanço")  
# 			st.write(dados6) 

# if a == 'Semifinais':
# 		dados1 = pd.read_excel('dados/R5outputSimulaçõesCopa(n=1000000).xlsx', index_col=0) 
# 		dados2 = pd.read_excel('dados/R5outputJogadoresArtilharia(n=1000000).xlsx', index_col=0) 
# 		dados3 = pd.read_excel('dados/R5outputFinaisMaisProvaveis(n=1000000).xlsx', index_col=0) 
# 		dados4 = pd.read_excel('dados/R5outputProbPorEtapa(n=1000000).xlsx', index_col=0)  
# 		dados6 = pd.read_excel('dados/R5outputAvançoPorEtapa.xlsx', index_col=0) 

# 		tab1, tab2, tab3, tab4, tab5 = st.tabs(["Simulações da Copa", 'Artilharia', "Finais Mais Prováveis",  'Probabilidades por Etapa', 'Probabilidades de Avanço'])
 
# 		with tab1:
# 			st.header("Simulações da Copa") 
# 			st.write(dados1)

# 		with tab2:  
# 			st.header("Previsões do Artilheiro")  
# 			st.write(dados2) 

# 		with tab3:  
# 			st.header("Finais Mais Prováveis")  
# 			st.write(dados3) 

# 		with tab4:  
# 			st.header("Probabilidades por Etapa")  
# 			st.write(dados4)  

# 		with tab5:  
# 			st.header("Probabilidades de Avanço")  
# 			st.write(dados6) 



# if a == 'Final':
# 		dados1 = pd.read_excel('dados/R6outputSimulaçõesCopa(n=1000000).xlsx', index_col=0) 
# 		dados2 = pd.read_excel('dados/R5outputJogadoresArtilharia(n=1000000).xlsx', index_col=0) 
# 		dados3 = pd.read_excel('dados/R6outputFinaisMaisProvaveis(n=1000000).xlsx', index_col=0) 
# 		dados4 = pd.read_excel('dados/R6outputProbPorEtapa(n=1000000).xlsx', index_col=0)  
# 		dados6 = pd.read_excel('dados/R6outputAvançoPorEtapa.xlsx', index_col=0) 

# 		tab1, tab2, tab3, tab4, tab5 = st.tabs(["Simulações da Copa", 'Artilharia', "Finais Mais Prováveis",  'Probabilidades por Etapa','Probabilidades de Avanço'])
 
# 		with tab1:
# 			st.header("Simulações da Copa") 
# 			st.write(dados1)

# 		with tab2:  
# 			st.header("Previsões do Artilheiro")  
# 			st.write(dados2) 

# 		with tab3:  
# 			st.header("Finais Mais Prováveis")  
# 			st.write(dados3) 

# 		with tab4:  
# 			st.header("Probabilidades por Etapa")  
# 			st.write(dados4)  

# 		with tab5:  
# 			st.header("Probabilidades de Avanço")  
# 			st.write(dados6) 