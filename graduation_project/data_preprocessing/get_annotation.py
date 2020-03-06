from bs4 import BeautifulSoup

xml_path = 'data/lidc/image/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192/069.xml'
with open(xml_path, 'r') as xml_file:
        markup = xml_file.read()
xml = BeautifulSoup(markup, features="xml")
print('ttt')

patient_id = xml.LidcReadMessage.ResponseHeader.SeriesInstanceUid.text

# 各专家对病例作出的诊断标示
reading_sessions = xml.LidcReadMessage.find_all('readingSession')

# 在每个病例里找结节信息， 包括 < unblindedReadNodule > 和 < nonNodule >
# < nonNodule > 是无关发现。
for reading_session in reading_sessions:
    nodules = reading_session.find_all("unblindedReadNodule") # 每个 unblindedReadNodule 表示一个（大或小）结节

    for nodule in nodules:
        if nodule.characteristics:
            characteristics_dic = {
                'subtlety': int(nodule.characteristics.subtlety.text),
                'internalStructure': int(nodule.characteristics.internalStructure.text),
                'calcification': int(nodule.characteristics.calcification.text),
                'sphericity': int(nodule.characteristics.sphericity.text),
                'margin': int(nodule.characteristics.margin.text),
                'lobulation': int(nodule.characteristics.lobulation.text),
                'spiculation': int(nodule.characteristics.spiculation.text),
                'texture': int(nodule.characteristics.texture.text),
                'malignancy': int(nodule.characteristics.malignancy.text),
                }
            nodule_id = nodule.noduleID.text
            print(characteristics_dic)

print('ttt')

