import re

def validar_cpf(cpf: str) -> bool:
    # Remove caracteres não numéricos
    cpf_numeros = re.sub(r'[^0-9]', '', cpf)

    # Verifica o tamanho e se é uma sequência repetida (ex: 111.111.111-11)
    if len(cpf_numeros) != 11 or cpf_numeros == cpf_numeros[0] * 11:
        return False

    # Calcula o primeiro dígito verificador
    soma = sum(int(cpf_numeros[i]) * (10 - i) for i in range(9))
    resto = (soma * 10) % 11
    if resto == 10:
        resto = 0
    if resto != int(cpf_numeros[9]):
        return False

    # Calcula o segundo dígito verificador
    soma = sum(int(cpf_numeros[i]) * (11 - i) for i in range(10))
    resto = (soma * 10) % 11
    if resto == 10:
        resto = 0
    if resto != int(cpf_numeros[10]):
        return False

    return True

def extrair_entidades_regex(texto_usuario: str, objeto_ner: dict) -> dict:
    # Garante que a chave parameters exista (inicia como lista por padrão)
    if "parameters" not in objeto_ner:
        objeto_ner["parameters"] = []

    # 1. Busca por CPFs e Placas
    padrao_cpf = r'\b\d{3}[\.]?\d{3}[\.]?\d{3}[-]?\d{2}\b'
    cpfs_encontrados = re.findall(padrao_cpf, texto_usuario)

    padrao_placa = r'\b[A-Za-z]{3}[- ]?[0-9][A-Za-z0-9][0-9]{2}\b'
    placas_encontradas = re.findall(padrao_placa, texto_usuario)

    # 2. Inserção baseada no tipo do objeto parameters (Lista ou Dict)
    if isinstance(objeto_ner["parameters"], list):
        # Como o erro indicou que é uma lista, faremos o append direto
        for cpf in cpfs_encontrados:
            objeto_ner["parameters"].append({
                "entity": cpf,
                "label": "cpf",
                "valid_cpf": validar_cpf(cpf) # Usa a função de validação criada anteriormente
            })
        
        for placa in placas_encontradas:
            placa_limpa = re.sub(r'[- ]', '', placa).upper()
            objeto_ner["parameters"].append({
                "label" : "placa_veiculo",
                "entity": placa_limpa,
            })
            
    elif isinstance(objeto_ner["parameters"], dict):
        # Fallback de segurança caso alguma intent retorne um dicionário
        if cpfs_encontrados and "cpf" not in objeto_ner["parameters"]:
            objeto_ner["parameters"]["cpf"] = []
        for cpf in cpfs_encontrados:
            objeto_ner["parameters"]["cpf"].append({
                "value": cpf,
                "valid_cpf": validar_cpf(cpf)
            })

        if placas_encontradas and "placa_veiculo" not in objeto_ner["parameters"]:
            objeto_ner["parameters"]["placa_veiculo"] = []
        for placa in placas_encontradas:
            placa_limpa = re.sub(r'[- ]', '', placa).upper()
            objeto_ner["parameters"]["placa_veiculo"].append({
                "value": placa_limpa
            })

    return objeto_ner