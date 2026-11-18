Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 5 items.

=== ITEM id=195d640079bbc686 ===
Gold sentences and facts:
Sentence 0: The culture of Australia is a Western culture , derived primarily from Britain but also influenced by the unique geography of Australia , the diverse input of Aboriginal , Torres Strait Islander and other Australian people . The British colonization of Australia began in 1788 , and waves of multi-ethnic migration followed . Evidence of a significant Anglo - Celtic heritage includes the predominance of the English language , the existence of a democratic system of government drawing upon the British traditions of Westminster Government , Parliamentarianism and constitutional monarchy , American constitutionalist and federalist traditions , Christianity as the dominant religion , and the popularity of sports originating in ( or influenced by ) the British Isles . Australian culture has diverged significantly since British settlement .
  Fact 0.0: The culture of Australia is a Western culture.
  Fact 0.1: The culture of Australia is derived primarily from Britain.
  Fact 0.2: The culture of Australia is also influenced by the unique geography of Australia.
  Fact 0.3: The culture of Australia is influenced by the diverse input of Aboriginal, Torres Strait Islander and other Australian people.
  Fact 0.4: British colonization of Australia began in 1788.
  Fact 0.5: Waves of multi-ethnic migration followed the British colonization of Australia.
  Fact 0.6: Evidence of a significant Anglo-Celtic heritage in Australia includes the predominance of the English language.
  Fact 0.7: Evidence of Anglo-Celtic heritage in Australia includes a democratic system of government drawing upon the British traditions of Westminster Government, Parliamentarianism and constitutional monarchy.
  Fact 0.8: Evidence of Anglo-Celtic heritage in Australia includes American constitutionalist and federalist traditions.
  Fact 0.9: Evidence of Anglo-Celtic heritage in Australia includes Christianity as the dominant religion.
  Fact 0.10: Evidence of Anglo-Celtic heritage in Australia includes the popularity of sports originating in (or influenced by) the British Isles.
  Fact 0.11: Australian culture has diverged significantly since British settlement.

Retrieved items:
<item 1>
Culture of Australia
The colonies established elected parliaments and rights for workers and women before most other Western nations . Federation in 1901 evidenced a growing sense
of national identity that had developed over the latter half of the 19th century , as seen in the works of the Heidelberg School painters and writers like Banjo
Paterson , Henry Lawson and Dorothea Mackellar . The World Wars profoundly altered Australia 's sense of identity , with World War I introducing the ANZAC
legend , and World War II seeing a reorientation from Britain to the United States as the nation 's foremost major ally . After the second war , 6.5 million
migrants from 200 nations brought immense new diversity , and Australians grew increasingly aware of their proximity to Asia . Over time , the diverse food ,
lifestyle and cultural practices of immigrants have been absorbed into mainstream Australian culture . Contents ( hide ) 1 Development of Australian culture 2
Symbols 3 Language 4 Humour 5 Arts 5.1 Literature 5.2 Theatre 5.3 Architecture 5.4 Visual arts 5.5 Cinema 5.6 Music 5.6. 1 Indigenous music 5.6.
</item 1>

<item 2>
Culture of Australia
A 1967 referendum to include all Aborigines in the national electoral roll census was overwhelmingly approved by voters . In 1984 , a group of Pintupi people
who were living a traditional hunter - gatherer desert - dwelling life were tracked down in the Gibson Desert and brought into a settlement . They are believed
to have been the last uncontacted tribe . While the British cultural influence remained strong into the 21st century , other influences became increasingly
important . The Hawaiian sport of surfing was adopted in Australia where a beach culture and the locally developed surf lifesaving movement was already
burgeoning in the early 20th century . American pop culture and cinema were embraced in the 20th century , with country music and later rock and roll sweeping
Australia , aided by the new technology of television and a host of American content . The 1956 Melbourne Olympics announced a confident , prosperous post-war
nation , and new cultural icons like Australian country music star Slim Dusty and dadaist Barry Humphries expressed a uniquely Australian identity . Australia
's contemporary immigration program has two components : a program for skilled and family migrants and a humanitarian program for refugees and asylum seekers .
</item 2>

<item 3>
Culture of Australia
Aboriginal musicians have turned their hand to Western popular musical forms , often to considerable commercial success . Pioneers include Lionel Rose and Jimmy
Little , while notable contemporary examples include Archie Roach , Kev Carmody , the Warumpi Band , Troy Cassar - Daley and Yothu Yindi . Geoffrey Gurrumul
Yunupingu ( formerly of Yothu Yindi ) has attained international success singing contemporary music in English and in the language of the Yolngu . Christine Anu
is a successful Torres Strait Islander singer . Amongst young Australian aborigines , African - American and Aboriginal hip hop music and clothing is popular .
The Deadly Awards are an annual celebration of Aboriginal and Torres Strait Islander achievement in music , sport , entertainment and community . Folk music and
national songs ( edit ) Cover of Old Bush Songs , Banjo Paterson 's 1905 collection of bush ballads The national anthem of Australia is `` Advance Australia
Fair '' .
</item 3>

<item 4>
Culture of Australia
By 2010 , the post-war immigration program had received more than 6.5 million migrants from every continent . The population tripled in the six decades to
around 21 million in 2010 , including people originating from 200 countries . More than 43 per cent of Australians were either born overseas or have one parent
who was born overseas . The population is highly urbanised , with more than 75 % of Australians living in urban centres , largely along the coast . Contemporary
Australia is a pluralistic society , rooted in liberal democratic traditions and espousing informality and egalitarianism as key societal values . While
strongly influenced by Anglo - Celtic origins , the culture of Australia has also been shaped by multi-ethnic migration which has influenced all aspects of
Australian life , including business , the arts , cuisine , sense of humour and sporting tastes . Contemporary Australia is also a culture that is profoundly
influenced by global movements of meaning and communication , including advertising culture . In turn , globalizing corporations from Holden to Exxon have
attempted to associate their brand with Australian cultural identity . This process intensified from the 1970s onwards .
</item 4>

<item 5>
Culture of Australia
These wars profoundly affected Australia 's sense of nationhood and a proud military legend developed around the spirit of Australia 's ANZAC troops , who came
to symbolise the virtues of mateship , courage and endurance for the nation . The Australian colonies had a period of extensive multi-ethnic immigration during
the Australian gold rushes of the latter half of the 19th century , but following Federation in 1901 , the Parliament instigated the White Australia Policy that
gave preference to British migrants and ensured that Australia remained a predominantly Anglo - Celtic society until well into the 20th Century . The post-World
War II immigration program saw the policy dismantled by successive governments , permitting large numbers of Southern European , and later Asian and Middle
Eastern migrants to arrive . The Menzies Government ( 1949 - 1966 ) and Holt Government dismantled the legal barriers to multi-ethnic immigration and by the
1970s , the Whitlam and Fraser Governments were promoting multiculturalism . Countries of birth of Australian estimated resident population , 2006 Some States
and Territories of Australia retained discriminatory laws relating to voting rights for Aborigines into the 1960s , at which point full legal equality was
established .
</item 5>

<item 6>
Culture of Australia
This process intensified from the 1970s onwards . According to Paul James , `` this consciously created interlock of the image of the multinational corporation
with aspects of Australia 's national mythology , has itself over the last decade ( the 1970s to 1980s ) contributed to the maintenance and evolution of those
national themes . But paradoxically , during the same period it has reinforced Australia 's international orientation . '' Symbols ( edit ) Main articles :
National symbols of Australia and Australian royal symbols Golden Wattle , Australia 's floral emblem and the source of Australia 's national colours , green
and gold When the Australian colonies federated on 1 January 1901 , an official competition for a design for an Australian flag was held . The design that was
adopted contains the Union Flag in the left corner , symbolising Australia 's historical links to the United Kingdom , the stars of the Southern Cross on the
right half of the flag indicating Australia 's geographical location , and the seven - pointed Federation Star in the bottom left representing the six states
and the territories of Australia . Other official flags include the Australian Aboriginal Flag , the Torres Strait Islander Flag and the flags of the individual
states and territories .
</item 6>

<item 7>
Culture of Australia
The British colonization of Australia began in 1788 , and waves of multi-ethnic migration followed . Evidence of a significant Anglo - Celtic heritage includes
the predominance of the English language , the existence of a democratic system of government drawing upon the British traditions of Westminster Government ,
Parliamentarianism and constitutional monarchy , American constitutionalist and federalist traditions , Christianity as the dominant religion , and the
popularity of sports originating in ( or influenced by ) the British Isles . Australian culture has diverged significantly since British settlement . Aboriginal
people are believed to have arrived as early as 60,000 years ago , and evidence of Aboriginal art in Australia dates back at least 30,000 years . Several states
and territories had their origins as penal colonies , with the first British convicts arriving at Sydney Cove in 1788 . Stories of outlaws like the bushranger
Ned Kelly have endured in Australian music , cinema and literature . The Australian gold rushes from the 1850s brought wealth as well as new social tensions to
Australia , including the miners ' Eureka Stockade rebellion . The colonies established elected parliaments and rights for workers and women before most other
Western nations .
</item 7>

<item 8>
Culture of Australia
During the 1990s , a new crop of Australian stars were successful in Hollywood , including Russell Crowe , Cate Blanchett and Heath Ledger . Between 1996 and
2013 , Catherine Martin won four Academy Awards for her costume and production designs , the most for any Australian . Saw ( 2004 ) and Wolf Creek ( 2005 ) are
credited with the revival of Australian horror . The domestic film industry is also supported by US producers who produce in Australia following the decision by
Fox head Rupert Murdoch to utilise new studios in Melbourne and Sydney where filming could be completed well below US costs . Notable productions include The
Matrix , Star Wars episodes II and III , and Australia starring Nicole Kidman and Hugh Jackman . Music ( edit ) Main article : Music of Australia Indigenous
music ( edit ) Main article : Indigenous Australian music Didgeridoo performers Music is an integral part of Aboriginal culture . The most famous feature of
their music is the didgeridoo . This wooden instrument , used amongst the Aboriginal tribes of northern Australia , makes a distinctive droning sound and it has
been adopted by a wide variety of non-Aboriginal performers .
</item 8>

<item 9>
Culture of Australia
Culture of Australia - Wikipedia Culture of Australia Jump to : navigation , search Part of a series on the Culture of Australia History People ( show )
Australian people African Australian Albanian Australian American Australian Anglo - Celtic Australian Asian Australian Assyrian Australian Bosnian Australian
Chinese Australian Cornish Australian English Australian European Australian Filipino Australian Greek Australian Indian Australian Indigenous Australian Irish
Australian Italian Australian Japanese Australian Jewish Australian Korean Australian Latin American Australian Lebanese Australian Macedonian Australian
Pakistani Australian Scottish Australian Serbian Australian South African Australian Sudanese Australian Taiwanese Australian Welsh Australian Languages
Traditions ( show ) Humour Mythology and folklore ( show ) Mythology folklore Cuisine Festivals Religion ( show ) Aboriginal Christianity ( Roman Catholic
Anglican Uniting Church Presbyterian Baptist Lutheran Coptic Orthodox more ) Buddhism Islam Sikhism Judaism Hinduism Bahá'í Faith Irreligion Art ( show )
Architecture Literature ( show ) Comic books Music and performing arts ( show ) Music Performing arts Media ( show ) Radio Television Cinema Sport Monuments (
show ) World Heritage Sites Symbols ( show ) Flag Coat of arms Australia portal The culture of Australia is a Western culture , derived primarily from Britain
but also influenced by the unique geography of Australia , the diverse input of Aboriginal , Torres Strait Islander and other Australian people .
</item 9>

<item 10>
Culture of Australia
Most Aboriginal Australians have a belief system based on the Dreaming , or Dreamtime , which refers both to a time when ancestral spirits created land and
culture , and to the knowledge and practices that define individual and community responsibilities and identity . Conflict and reconciliation between indigenous
and non-indigenous Australians has been a source of much art and literature in Australia , and ancient Aboriginal artistic styles and iconic inventions such as
the boomerang , the didgeridoo and Indigenous Australian music have become symbols of modern Australia . Governor Arthur Phillip hoists the British flag over
the new colony at Sydney in 1788 South Australian suffragette Catherine Helen Spence ( 1825 -- 1910 ) . The Australian colonies established democratic
parliaments from the 1850s and began to grant women the vote in the 1890s . The arrival of the first British settlers at what is now Sydney in 1788 introduced
Western civilisation to the Australian continent .
</item 10>
=== END ITEM id=195d640079bbc686 ===

=== ITEM id=81b45a497df6dc0d ===
Gold sentences and facts:
Sentence 0: Beijing ( / beɪˈdʒɪŋ / ; Mandarin : ( pèi. tɕíŋ ) ( listen ) ) , formerly romanized as Peking , is the capital of the People 's Republic of China , the world 's second most populous city proper , and most populous capital city . The city , located in northern China , is governed as a direct - controlled municipality under the national government with 16 urban , suburban , and rural districts . Beijing Municipality is surrounded by Hebei Province with the exception of neighboring Tianjin Municipality to the southeast ; together the three divisions form the Jingjinji metropolitan region and the national capital region of China .
  Fact 0.0: Beijing is the capital of the People's Republic of China.
  Fact 0.1: Beijing was formerly romanized as Peking.
  Fact 0.2: Beijing is the world's second most populous city proper.
  Fact 0.3: Beijing is the most populous capital city.
  Fact 0.4: Beijing is located in northern China.
  Fact 0.5: Beijing is governed as a direct-controlled municipality under the national government.
  Fact 0.6: Beijing has 16 urban, suburban, and rural districts.
  Fact 0.7: Beijing Municipality is surrounded by Hebei Province, except for the neighboring Tianjin Municipality to the southeast.
  Fact 0.8: Beijing Municipality, Hebei Province, and Tianjin Municipality together form the Jingjinji metropolitan region.
  Fact 0.9: Beijing Municipality, Hebei Province, and Tianjin Municipality together form the national capital region of China.
Sentence 1: Over the past 3,000 years , the city of Beijing has had numerous other names . The name Beijing , which means `` Northern Capital '' ( from the Chinese characters 北 for north and 京 for capital ) , was applied to the city in 1403 during the Ming dynasty to distinguish the city from Nanjing ( the `` Southern Capital '' ) . The English spelling is based on the pinyin romanization of the two characters as they are pronounced in Standard Mandarin . An older English spelling , Peking , is the postal romanization of the same two characters as they are pronounced in Chinese dialects spoken in the southern port towns first visited by European traders and missionaries . Those dialects preserve the Middle Chinese pronunciation of 京 as kjaeng , prior to a phonetic shift in the northern dialects to the modern pronunciation . Although Peking is no longer the common name for the city , some of the city 's older locations and facilities , such as Beijing Capital International Airport , with IATA Code PEK , and Peking University , still use the former romanization .
  Fact 1.0: Over the past 3,000 years, the city of Beijing has had numerous other names.
  Fact 1.1: The name Beijing means 'Northern Capital'.
  Fact 1.2: The name Beijing is made of the Chinese characters 北 (north) and 京 (capital).
  Fact 1.3: The name Beijing was applied to the city in 1403 during the Ming dynasty.
  Fact 1.4: The name Beijing was applied to distinguish the city from Nanjing, the 'Southern Capital'.
  Fact 1.5: The English spelling Beijing is based on the pinyin romanization of the two characters as pronounced in Standard Mandarin.
  Fact 1.6: The older English spelling Peking is the postal romanization of the same two characters as pronounced in Chinese dialects spoken in the southern port towns.
  Fact 1.7: The southern port towns were the first visited by European traders and missionaries.
  Fact 1.8: The Chinese dialects spoken in the southern port towns preserve the Middle Chinese pronunciation of 京 as kjaeng.
  Fact 1.9: That pronunciation of 京 as kjaeng comes before a phonetic shift in the northern dialects to the modern pronunciation.
  Fact 1.10: Peking is no longer the common name for the city.
  Fact 1.11: Some of the city's older locations and facilities still use the former romanization Peking.
  Fact 1.12: Beijing Capital International Airport, with IATA code PEK, still uses the former romanization.
  Fact 1.13: Peking University still uses the former romanization.

Retrieved items:
<item 1>
Beijing
Retrieved 22 July 2009 . External links Find more aboutBeijingat Wikipedia 's sister projects Definitions from Wiktionary Media from Wikimedia Commons News from
Wikinews Quotations from Wikiquote Texts from Wikisource Textbooks from Wikibooks Travel guide from Wikivoyage Learning resources from Wikiversity Economic
profile for Beijing at HKTDC Visit Beijing Facebook Page Photograph of The approach to Peking -- outside the walls taken in 1890 by Sir Henry Norman Preceded by
Lin'an ( Song dynasty ) Capital of China ( as Dadu of Yuan ) 1264 -- 1368 Succeeded by Nanjing ( Ming dynasty ) Preceded by Nanjing ( Ming dynasty ) Capital of
China 1420 -- 1928 Succeeded by Nanjing ( ROC ) Preceded by Nanjing ( ROC ) Capital of the People 's Republic of China 1949 -- present Succeeded by present
capital Places adjacent to Beijing Zhangjiakou , Hebei Chengde , Hebei Chengde , Hebei Zhangjiakou , Hebei Beijing Langfang , Hebei Baoding , Hebei Langfang ,
</item 1>

<item 2>
Beijing
1928 -- 1937 ; 1945 -- 1949 ) Literal meaning `` Northern Capital '' ( show ) Transcriptions Standard Mandarin Hanyu Pinyin Běijīng Bopomofo ㄅ ㄟ ˇ ㄐ ㄧ ㄥ Gwoyeu
Romatzyh Beeijing Wade -- Giles Pei - ching IPA ( pèi. tɕíŋ ) Wu Suzhounese Poh - cin Hakka Romanization Bet - gin Yue : Cantonese Yale Romanization Bākgìng or
Bākgīng IPA ( pɐ́k̚. kêŋ ) or ( pɐ́k̚. kéŋ ) Jyutping Bak1ging1 Southern Min Hokkien POJ Pak - kian Tâi - lô Pak - kiann Eastern Min Fuzhou BUC Báe̤k - gĭng
Beijing ( / beɪˈdʒɪŋ / ; Mandarin : ( pèi.
</item 2>

<item 3>
Beijing
The official Latin alphabet abbreviation for Beijing is `` BJ '' . History Main article : History of Beijing Early History The earliest traces of human
habitation in the Beijing municipality were found in the caves of Dragon Bone Hill near the village of Zhoukoudian in Fangshan District , where Peking Man lived
. Homo erectus fossils from the caves date to 230,000 to 250,000 years ago . Paleolithic Homo sapiens also lived there more recently , about 27,000 years ago .
Archaeologists have found neolithic settlements throughout the municipality , including in Wangfujing , located in downtown Beijing . The first walled city in
Beijing was Jicheng , the capital city of the state of Ji and was built in 1045 BC . Within modern Beijing , Jicheng was located around the present Guang'anmen
area in the south of Xicheng District . This settlement was later conquered by the state of Yan and made its capital . Early Imperial China The Tianning Pagoda
, built around 1120 during the Liao dynasty . After the First Emperor unified China , Jicheng became a prefectural capital for the region .
</item 3>

<item 4>
Beijing
tɕíŋ ) ( listen ) ) , formerly romanized as Peking , is the capital of the People 's Republic of China , the world 's second most populous city proper , and
most populous capital city . The city , located in northern China , is governed as a direct - controlled municipality under the national government with 16
urban , suburban , and rural districts . Beijing Municipality is surrounded by Hebei Province with the exception of neighboring Tianjin Municipality to the
southeast ; together the three divisions form the Jingjinji metropolitan region and the national capital region of China . As a city combining both modern and
traditional architecture , Beijing is a megacity rich in history , exemplified in its global influence in politics , economy , education , history , culture ,
and technology . Beijing is the second largest Chinese city by urban population after Shanghai and is the nation 's political , cultural , and educational
center . It is home to the headquarters of most of China 's largest state - owned companies and is a major hub for the national highway , expressway , railway ,
and high - speed rail networks .
</item 4>

<item 5>
Beijing
The name Beijing , which means `` Northern Capital '' ( from the Chinese characters 北 for north and 京 for capital ) , was applied to the city in 1403 during the
Ming dynasty to distinguish the city from Nanjing ( the `` Southern Capital '' ) . The English spelling is based on the pinyin romanization of the two
characters as they are pronounced in Standard Mandarin . An older English spelling , Peking , is the postal romanization of the same two characters as they are
pronounced in Chinese dialects spoken in the southern port towns first visited by European traders and missionaries . Those dialects preserve the Middle Chinese
pronunciation of 京 as kjaeng , prior to a phonetic shift in the northern dialects to the modern pronunciation . Although Peking is no longer the common name for
the city , some of the city 's older locations and facilities , such as Beijing Capital International Airport , with IATA Code PEK , and Peking University ,
still use the former romanization . The single Chinese character abbreviation for Beijing is 京 , which appears on automobile license plates in the city . The
official Latin alphabet abbreviation for Beijing is `` BJ '' .
</item 5>

<item 6>
Beijing
And under the Hongwu Emperor 's feudal policies it was given to Zhu Di , one of his sons , who was created `` Prince of Yan '' . The early death of Zhu
Yuanzhang 's heir led to a succession struggle on his death , one that ended with the victory of Zhu Di and the declaration of the new Yongle era . Since his
harsh treatment of the Ming capital Yingtian ( modern Nanjing ) alienated many there , he established his fief as a new co-capital . The city of Beiping became
Beijing ( northern capital ) or Shuntian in 1403 . The construction of the new imperial residence , the Forbidden City , took from 1406 to 1420 ; this period
was also responsible for several other of the modern city 's major attractions , such as the Temple of Heaven and Tian'anmen . On 28 October 1420 , the city was
officially designated the capital of the Ming dynasty in the same year that the Forbidden City was completed . Beijing became the empire 's primary capital and
Yingtian , also called Nanjing ( southern capital ) , became the co-capital .
</item 6>

<item 7>
Beijing
( A 1425 order by Zhu Di 's son , the Hongxi Emperor , to return the primary capital to Nanjing was never carried out : he died , probably of a heart attack ,
the next month . He was buried , like almost every Ming emperor to follow him , in an elaborate necropolis to Beijing 's north . ) By the 15th century , Beijing
had essentially taken its current shape . The Ming city wall continued to serve until modern times , when it was pulled down and the 2nd Ring Road was built in
its place . It is generally believed that Beijing was the largest city in the world for most of the 15th , 16th , 17th , and 18th centuries . The first known
church was constructed by Catholics in 1652 at the former site of Matteo Ricci 's chapel ; the modern Nantang Cathedral was later built upon the same site . The
capture of Beijing by Li Zicheng 's peasant army in 1644 ended the dynasty , but he and his Shun court abandoned the city without a fight when the Manchu army
of Prince Dorgon arrived 40 days later .
</item 7>

<item 8>
Beijing
1 Towns 4.1. 2 Neighborhoods 4.2 Judiciary and procuracy 4.3 Diplomatic missions 5 Economy 5.1 Sector composition 5.2 Economic zones 6 Demographics 6.1
Metropolitan area 7 Culture 7.1 Places of interest 8 Religion 8.1 Chinese folk religion and Taoism 8.2 Buddhism 8.3 Islam 8.4 Christianity 9 Media 9.1
Television and radio 9.2 Press 10 Sports 10.1 Events 10.2 Venues 10.3 Clubs 11 Transportation 11.1 Rail and high - speed rail 11.2 Roads and expressways 11.3
Air 11.4 Public transit 11.5 Taxi 11.6 Bicycles 12 Defense and aerospace 13 Nature and wildlife 14 See also 15 References 16 Further reading 17 External links
Etymology See also : Names of Beijing Over the past 3,000 years , the city of Beijing has had numerous other names .
</item 8>

<item 9>
Beijing
In 938 , after the fall of the Tang , the Later Jin ceded the entire northern frontier to the Khitan Liao dynasty , which treated the city as Nanjing , or the
`` Southern Capital '' , one of four secondary capitals to complement its `` Supreme Capital '' , Shangjing ( modern Baarin Left Banner in Inner Mongolia ) .
Some of the oldest surviving structures in Beijing date to the Liao period , including the Tianning Pagoda . Miaoying Temple and its White Stupa in Dadu which
was built by Kublai Khan The Liao fell to the Jurchen Jin dynasty in 1122 , which gave the city to the Song dynasty and then retook it in 1125 during its
conquest of northern China . In 1153 , the Jurchen Jin made Beijing their `` Central Capital '' , or Zhongdu . The city was besieged by Genghis Khan 's invading
Mongolian army in 1213 and razed to the ground two years later .
</item 9>

<item 10>
Beijing
Beijing - wikipedia Beijing Jump to : navigation , search `` Peking '' redirects here . For other uses , see Beijing ( disambiguation ) and Peking (
disambiguation ) . Beijing 北京 市 Municipality Beijing Municipality Clockwise from top : Beijing CBD skyline , Tiananmen , Temple of Heaven , National Center for
the Performing Arts , and Beijing National Stadium Location of Beijing Municipality within China Coordinates : 39 ° 55 ′ N 116 ° 23 ′ E ﻿ / ﻿ 39.917 ° N 116.383
° E ﻿ / 39.917 ; 116.383 Coordinates : 39 ° 55 ′ N 116 ° 23 ′ E ﻿ / ﻿ 39.917 ° N 116.383 ° E ﻿ / 39.917 ; 116.383 Country China Established date 1045 BCE ( Zhou
Dynasty ) Divisions - County - level - Township - level 16 districts 289 towns and villages Government Type Municipality Party Secretary Cai Qi Mayor Chen
Jining Congress Chairman Li Wei ( zh ) Conference Chairman Ji Lin Area Municipality 16,411 km ( 6,336 sq mi ) Land 16,
</item 10>
=== END ITEM id=81b45a497df6dc0d ===

=== ITEM id=c4af9b5319f962b3 ===
Gold sentences and facts:
Sentence 0: Downton Abbey is a British period drama television series created by Julian Fellowes and co-produced by Carnival Films and Masterpiece . It first aired on ITV in the United Kingdom on 26 September 2010 , and on PBS in the United States on 9 January 2011 , as part of the Masterpiece Classic anthology . Six series have been made , the sixth airing in the autumn of 2015 in the UK and Ireland , and in January 2016 in the United States . On 26 March 2015 , the sixth series was confirmed to be the final series , with the final episode airing in the UK on 25 December 2015 on ITV . During the course of the programme , 52 episodes of Downton Abbey aired over six series .
  Fact 0.0: Downton Abbey is a British period drama television series.
  Fact 0.1: Downton Abbey was created by Julian Fellowes.
  Fact 0.2: Downton Abbey was co-produced by Carnival Films and Masterpiece.
  Fact 0.3: Downton Abbey first aired on ITV in the United Kingdom on 26 September 2010.
  Fact 0.4: Downton Abbey first aired on PBS in the United States on 9 January 2011.
  Fact 0.5: The PBS airing was part of the Masterpiece Classic anthology.
  Fact 0.6: Six series of Downton Abbey have been made.
  Fact 0.7: The sixth series aired in the autumn of 2015 in the UK and Ireland.
  Fact 0.8: The sixth series aired in January 2016 in the United States.
  Fact 0.9: On 26 March 2015, the sixth series was confirmed to be the final series.
  Fact 0.10: The final episode of Downton Abbey aired in the UK on 25 December 2015 on ITV.
  Fact 0.11: 52 episodes of Downton Abbey aired over six series.

Retrieved items:
<item 1>
List of Downton Abbey episodes
45 Christmas special No . overall Title Directed by Written by Original air date UK viewers ( millions ) 16 `` Christmas at Downton Abbey '' Brian Percival
Julian Fellowes 25 December 2011 ( 2011 - 12 - 25 ) 12.11 Series 3 ( 2012 ) ( edit ) ITV confirmed the commissioning of a third series on 3 November 2011 . It
entered production in early 2012 , and began airing 16 September 2012 on ITV and aired in the United States beginning 6 January 2013 . Series 3 No . overall No.
in series Title Directed by Written by Original air date UK viewers ( millions ) 17 `` Episode One '' Brian Percival Julian Fellowes 16 September 2012 ( 2012 -
09 - 16 ) 11.43 7.9 ( PBS ) March / April 1920 . Cora 's mother , Martha Levinson , sweeps into Downton for Matthew and Mary 's wedding , aggravating the
Dowager Countess ( among others ) with her critical and `` revolutionary '' American way of seeing things .
</item 1>

<item 2>
List of Downton Abbey episodes
overall No. in series Title Directed by Written by Original air date UK viewers ( millions ) 26 `` Episode One '' David Evans Julian Fellowes 22 September 2013
( 2013 - 09 - 22 ) 11.96 February 1922 . Six months on , a distraught Lady Mary mourns Matthew , and feels little motherly love for baby George , now six months
old . Lord Grantham and Branson dispute the management of the estate and struggle to pay death duties . Miss O'Brien suddenly leaves to serve Lady Flintshire ,
on her way to India . Needing a maid in a hurry , Cora hires former maid Edna , who has trained as a lady 's maid , on the strength of the reference which Tom
persuaded Mrs Hughes to give her ; Carson and Mrs Hughes see trouble ahead , and warn Tom to keep his distance . Miss West , the nanny of George and Sybbie ,
Mary 's son and Tom 's daughter , respectively , acts insolently with Thomas , and the underbutler lets Cora know of her tendency to leave them to their own
devices .
</item 2>

<item 3>
List of Downton Abbey episodes
68 8 ( + 1 ) 16 September 2012 ( 2012 - 09 - 16 ) 4 November 2012 ( 2012 - 11 - 04 ) 25 December 2012 ( special ) 11.91 8 ( + 1 ) 22 September 2013 ( 2013 - 09
- 22 ) 10 November 2013 ( 2013 - 11 - 10 ) 25 December 2013 ( special ) 11.84 5 8 ( + 1 ) 21 September 2014 ( 2014 - 09 - 21 ) 9 November 2014 ( 2014 - 11 - 09
) 25 December 2014 ( special ) 10.40 6 8 ( + 1 ) 20 September 2015 ( 2015 - 09 - 20 ) 8 November 2015 ( 2015 - 11 - 08 ) 25 December 2015 ( special ) 10.
</item 3>

<item 4>
List of Downton Abbey episodes
Christmas special No . overall Title Directed by Written by Original air date UK viewers ( millions ) 34 `` The London Season '' Jon East Julian Fellowes 25
December 2013 ( 2013 - 12 - 25 ) 9.4 August 1923 : The Granthams move to their London residence for Rose 's debutante ball and presentation at court , bringing
most servants . They are joined by Cora 's brother , Harold , and mother , Martha , from the United States . Lord Gillingham and Mr Blake attend , both still
courting Mary , and mutually antagonistic . Tom and Thomas initially remain at Downton . Tom unexpectedly meets with Sarah Bunting , who asks for a tour of
Downton Abbey . When they go to the upstairs gallery , Thomas sees them and later in London tells Lord Grantham about the incident , insinuating Tom took Sarah
to a bedroom . Tom later tells Lord Grantham the truth of the matter .
</item 4>

<item 5>
List of Downton Abbey episodes
List of Downton Abbey episodes - Wikipedia List of Downton Abbey episodes Jump to : navigation , search This article 's plot summaries may be too long or
excessively detailed . Please help improve them by removing unnecessary details and making them more concise . ( January 2016 ) ( Learn how and when to remove
this template message ) Downton Abbey is a British period drama television series created by Julian Fellowes and co-produced by Carnival Films and Masterpiece .
It first aired on ITV in the United Kingdom on 26 September 2010 , and on PBS in the United States on 9 January 2011 , as part of the Masterpiece Classic
anthology . Six series have been made , the sixth airing in the autumn of 2015 in the UK and Ireland , and in January 2016 in the United States . On 26 March
2015 , the sixth series was confirmed to be the final series , with the final episode airing in the UK on 25 December 2015 on ITV . During the course of the
programme , 52 episodes of Downton Abbey aired over six series .
</item 5>

<item 6>
List of Downton Abbey episodes
Molesley and Baxter take a photo of Bates to different pubs in York , and find the one where he had lunch on the day in question , proving he was not in London
when Green was killed . The witness who identified Anna becomes uncertain , and she is released on bail . A big Christmas party is held for the Downton
household . It also serves as a goodbye party for Tom , who has found work as a writer in America and will be taking his daughter Sybbie with him . Bates
returns and enjoys a tender reunion with Anna . During a private break , Violet confesses to Isobel that she and Prince Kuragan had attempted to run away
together . Series 6 ( 2015 ) ( edit ) The sixth series premiered on 20 September 2015 in the UK and on 3 January 2016 in the United States . This series was the
last , the 2015 Christmas special titled `` The Finale '' . Due to increasingly popular ratings worldwide , a film is said to be in production . Series 6 No .
overall No.
</item 6>

<item 7>
List of Downton Abbey episodes
Christmas special No . overall Title Directed by Written by Original air date UK viewers ( millions ) 52 `` The Finale '' Michael Engler Julian Fellowes 25
December 2015 ( 2015 - 12 - 25 ) 11.22 29 September -- 31 December 1925 : Carson fumbles serving dinner when his hands begin shaking , suffering from an
unspecified hereditary disease his family called ' the palsy ' . Thomas secures a job with an elderly couple with a staff of only three , but is unhappy there .
Edith offers to expand Spratt 's magazine column to a whole page ; Denker discovers Spratt 's secret , and tries to get him dismissed by revealing it to the
Dowager Countess , but she finds the column jolly amusing and keeps him on . Molesley leaves service when the Headmaster offers him a teaching position , plus a
cottage ; upon parting , he and Baxter finally express their mutual romantic affection . Lord Merton tells Isobel he suffers from pernicious anaemia , but still
wants to marry her .
</item 7>

<item 8>
List of Downton Abbey episodes
33 11 `` Episode Four '' Brian Kelly Julian Fellowes 9 October 2011 ( 2011 - 10 - 09 ) 11.30 12 5 `` Episode Five '' Brian Kelly Julian Fellowes 16 October 2011
( 2011 - 10 - 16 ) 11.59 13 6 `` Episode Six '' Andy Goddard Julian Fellowes 23 October 2011 ( 2011 - 10 - 23 ) 11.33 14 7 `` Episode Seven '' James Strong
Julian Fellowes 30 October 2011 ( 2011 - 10 - 30 ) 12.26 15 8 `` Episode Eight '' James Strong Julian Fellowes 6 November 2011 ( 2011 - 11 - 06 ) 12.45
Christmas special No .
</item 8>

<item 9>
List of Downton Abbey episodes
With the future of Downton hanging in the balance , Mary and the Dowager Countess conspire to show Downton at its grandest to persuade Mrs Levinson to give
Downton more money and save them from financial ruin . Their attempts are impeded by the oven breaking down , giving Mrs Levinson a chance to take over running
the party herself in radical fashion . She reveals , when finally asked for help , that though she will `` entertain them in New York '' she can not provide
them with money , as her own late husband tied it up in the estate , much as Violet 's did . She tells them they have to adapt to the changing world , and
states her intent to return to America . Meanwhile downstairs , O'Brien and Thomas wage war against one another due to Alfred 's recent promotion to valet to
Matthew , whilst Mrs Levinson 's maid takes a shine to Alfred . Mrs Hughes faces a cancer scare that she shares only with Mrs Patmore and Doctor Clarkson , and
Isobel Crawley finds that Ethel has become a prostitute .
</item 9>

<item 10>
List of Downton Abbey episodes
Series 6 No . overall No. in series Title Directed by Written by Original air date UK viewers ( millions ) 44 `` Episode One '' Minkie Spiro Julian Fellowes 20
September 2015 ( 2015 - 09 - 20 ) 10.32 April 1925 . Big changes are afoot . Robert anticipates having to cut costs and reduce staff , worrying Carson . Mrs
Hughes , nervous she will be embarrassed when performing her wifely duties after marrying Carson , asks Mrs Patmore to sound him out ; after much embarrassment
, Carson says he expects them to have a full marriage , which reassures Mrs Hughes . The local hospital is threatened with a take - over by the larger County
Hospital ; the board is split , with Violet objecting , and Isobel approving in the name of efficiency . Rita Bevan , a former chambermaid at the hotel where
Mary and Lord Gillingham stayed , attempts to blackmail Mary by exposing their tryst .
</item 10>
=== END ITEM id=c4af9b5319f962b3 ===

=== ITEM id=967223e1a5dd496e ===
Gold sentences and facts:
Sentence 0: A justice of the peace ( JP ) is a judicial officer , of a lower or puisne court , elected or appointed by means of a commission ( letters patent ) to keep the peace . In past centuries the term commissioner of the peace was often used with the same meaning . Depending on the jurisdiction , such justices dispense summary justice or merely deal with local administrative applications in common law jurisdictions . Justices of the peace are appointed or elected from the citizens of the jurisdiction in which they serve , and are ( or were ) usually not required to have any formal legal education in order to qualify for the office . Some jurisdictions have varying forms of training for JPs .
  Fact 0.0: A justice of the peace (JP) is a judicial officer.
  Fact 0.1: A justice of the peace is a judicial officer of a lower or puisne court.
  Fact 0.2: A justice of the peace is elected or appointed by means of a commission (letters patent).
  Fact 0.3: A justice of the peace is elected or appointed to keep the peace.
  Fact 0.4: In past centuries the term commissioner of the peace was often used with the same meaning as justice of the peace.
  Fact 0.5: Depending on the jurisdiction, justices of the peace dispense summary justice.
  Fact 0.6: Depending on the jurisdiction, justices of the peace merely deal with local administrative applications in common law jurisdictions.
  Fact 0.7: Justices of the peace are appointed or elected from the citizens of the jurisdiction in which they serve.
  Fact 0.8: Justices of the peace are (or were) usually not required to have any formal legal education in order to qualify for the office.
  Fact 0.9: Some jurisdictions have varying forms of training for justices of the peace.

Retrieved items:
<item 1>
Justice of the peace
Their judicial powers were replaced by full - time , legally qualified district justices ( now called district judges ) and their quasi-judicial powers by lay
Peace Commissioners . Peace commissioners may sign statutory declarations , and may issue summons and search warrants to the Garda Síochána ( Irish police ) . A
Peace Commissioner can not sign an Affidavit , in Ireland that function is performed by a Commissioner for Oaths . Jamaica ( edit ) A justice of the peace ( JP
) , according to the Ministry of Justice , is a person of unquestionable integrity who seeks to promote and protect the rights of the individual and helps to
provide justice to persons in a particular community . Additionally , the JP serves as a justice in petty court sessions , attends juvenile court sessions ,
issues summonses , considers applications for bail , explains and signs legal documents , sits on licensing panels , and gives counsel / advice . Any Jamaican
citizen that can speak and write English is eligible to become a JP . Any club / organisation / citizen can recommend someone to become JP for a community . JPs
are chosen under the Governor - General 's discretion .
</item 1>

<item 2>
Justice of the peace
United states ( edit ) In some US states , the justice of the peace is a judge of a court of limited jurisdiction , a magistrate , or a quasi-judicial official
with certain statutory or common law magisterial powers . Some states have special qualifications or unique features for the office . The justice of the peace
typically presides over a court that hears misdemeanor cases , traffic violations , and other petty criminal infractions . The justice of the peace may also
have authority over cases involving small debts , landlord and tenant disputes , or other small claims court proceedings . Proceedings before justices of the
peace are often faster and less formal than the proceedings in other courts . In some jurisdictions a party convicted or found liable before a justice of the
peace may have the right to a trial de novo before the judge of a higher court rather than an appeal strictly considered . A justice of the peace also performs
civil marriages . Arizona ( edit ) A justice of the peace has the same jurisdiction as a municipal magistrate with respect to traffic and misdemeanor cases and
restraining orders , though over cases whose affairs are not contained within the confines of a single municipality .
</item 2>

<item 3>
Justice of the peace
Administrative justice of the peace are court officers appointed by the Minister of Justice , and perform duties such as receiving criminal informations and
issuing warrants . Presiding justice of the peace are appointed by commission under the Great Seal , and can try some criminal matters and issue warrants . They
are appointed from advocates of at least ten years ' standing and serve full - time until the age of 70 . Yukon ( edit ) In Yukon , justices of the peace are
lay officers of the court . They sit in the Justice of the Peace Court , which is part of the Territorial Court of Yukon . Hong Kong ( edit ) In Hong Kong , the
historical functions of justices of the peace have been replaced by full - time , legally qualified magistrates . Nowadays , justices of the peace are
essentially titles of honour given by the Government to community leaders , and to certain officials while they are in their terms of offices . They have no
judicial functions , and their main duties include visiting prisons , institutions for young offenders and drug addicts , psychiatric hospitals , remand homes ,
places of refuge , reception and detention centres , administering statutory declarations , and serving as members of advisory panels .
</item 3>

<item 4>
In the United States, justices of the peace are officials with limited judicial or magisterial powers. They commonly handle misdemeanors, traffic offenses,
small claims, landlord–tenant disputes, and civil marriages; proceedings are often faster and less formal, and some cases may receive a new trial in a higher
court. In Arizona, elected JPs serve four-year terms by precinct and hear traffic, misdemeanor, restraining-order, county-ordinance, civil claims up to $10,000,
small claims up
</item 4>

<item 5>
Justices of the peace (JPs) originated in England to preserve the King’s peace; the title dates to 1361. Their roles vary widely: they may hear minor criminal
or civil cases, issue warrants, set bail, conduct inquests, administer oaths, witness documents, or perform marriages. Some are trained lawyers; many are lay or
volunteer officials. U.S. powers and election or appointment rules differ by state. In Britain, lay magistrates handle most summary offences; Scotland’s
</item 5>

<item 6>
Justice of the peace
Additionally , the Justice Court hears cases involving county ordinances ( ordinances enacted by the board of supervisors that apply only to unincorporated
areas ) , civil lawsuits up to a limit of $10,000 , small claims cases up to $2,500 , and issues evictions , called writs of restitution ( after a forcible
detainer or special detainer action ( eviction ) being successfully completed by a landlord ) . Justices of the peace , also called JPs , or Judges of the
Justice Court , are elected in partisan elections for four - year terms from specific districts called precincts . They have the same authority and
responsibility as all other judges in the state with respect to performing marriages , administrating oaths , adhering to the code of judicial conduct , and all
aspects of justice administration . However , Arizona law does not require justices of the peace to be lawyers . Many justices of the peace are not legally
trained , although all are required by the Arizona Supreme Court to complete a course at the Arizona Judicial College . As with JPs , municipal judges in
Arizona are not required to be lawyers . Arkansas ( edit ) This section does not cite any sources .
</item 6>

<item 7>
A justice of the peace (JP) is a lower-court judicial officer appointed or elected to maintain the peace. Depending on the jurisdiction, JPs handle summary
justice or local administrative matters. They are generally drawn from local citizens and often do not need formal legal education, though some jurisdictions
provide training. Historically, “commissioner of the peace” was a similar title.
</item 7>

<item 8>
Justice of the peace
Justice of the peace - wikipedia Justice of the peace Jump to : navigation , search This article needs additional citations for verification . Please help
improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( January 2017 ) ( Learn how and when to
remove this template message ) Taos County , New Mexico . Justice of the Peace Montoyo hears a minor case . ( 1941 ) A justice of the peace ( JP ) is a judicial
officer , of a lower or puisne court , elected or appointed by means of a commission ( letters patent ) to keep the peace . In past centuries the term
commissioner of the peace was often used with the same meaning . Depending on the jurisdiction , such justices dispense summary justice or merely deal with
local administrative applications in common law jurisdictions . Justices of the peace are appointed or elected from the citizens of the jurisdiction in which
they serve , and are ( or were ) usually not required to have any formal legal education in order to qualify for the office . Some jurisdictions have varying
forms of training for JPs . Contents ( hide ) 1 History 2 Modern use 2.1 Australia 2.1.
</item 8>

<item 9>
Justice of the peace
A JP has contempt power of $100 and up to 3 days in jail per occurrence . A JP is also authorized to perform marriage ceremonies . Replacement with other courts
( edit ) In many states , the office of justice of the peace has been abolished or transferred to another court , such as the magistrate court . In larger
cities , cases may be heard in a municipal court which has jurisdiction only within that city . Most efforts to abolish the office of justice of the peace have
been led by the American Bar Association , which views non-lawyer judges as no longer necessary , as there are now far more persons with formal legal education
than in the past when justices of the peace were first used . California formerly had justice of the peace courts staffed by lay judges , but began phasing them
out after a landmark 1974 decision in which the Supreme Court of California unanimously held that it was a violation of federal due process ( under the
Fourteenth Amendment to the U.S. Constitution ) to allow a non-lawyer to preside over a criminal trial which could result in incarceration of the defendant .
</item 9>

<item 10>
Justice of the peace
These courts were introduced in 2009 as a replacement for the district courts ( established in 1975 ) , which in turn replaced burgh police courts . Justices
sit alone or in threes with a qualified legal assessor as convener or clerk of court . They handle many cases of breaches of the peace , drunkenness , minor
assaults , petty theft , and offences under the Civic Government ( Scotland ) Act 1982 . The maximum sentencing power of a justice of the peace is 60 days
imprisonment , or a fine up to £ 2,500 , or both , and the ability to disqualify drivers . In 2006 , the Scottish Government announced its intention to unify
the management of the sheriff and district courts in Scotland , but retaining lay justices , as part of its initiative to create a unified judiciary under the
Lord President . Following the passage of the Criminal Proceedings etc . ( Reform ) ( Scotland ) Act 2007 the justice of the peace courts were implemented on a
sheriffdom - by - sheriffdom basis . In Glasgow , the volume of business required the employment of three solicitors as `` stipendiary magistrates '' who sat in
place of the lay justices .
</item 10>
=== END ITEM id=967223e1a5dd496e ===

=== ITEM id=dc64e3234bd1b571 ===
Gold sentences and facts:
Sentence 0: Barbara Eden as Stella Johnson , a widowed cosmetics salesperson with a young daughter , whose joie de vivre , her principles , and her individuality anger most of the powerful people in town and on the Parent Teachers Association board . She inevitably becomes PTA president after embarrassing Flora , the snobby former President and her equally snobby friends . Ronny Cox as Willis `` Will '' Newton , a wealthy member of the PTA who does n't agree with Flora and her snobby allies and is in love with Stella . He and Stella get married at the end of the movie . Nanette Fabray as Alice Finley , Stella 's best friend and co-conspirator against the snobs of Harper Valley . She owns and operates the La Moderne Beauty Shop . Louis Nye as Kirby Baker , a local real estate agent who is allied with Flora and her friends . Notorious for being a lecherous playboy and infamous for impregnating his secretaries , one of whom , Gladys Wilmot , left town to see an abortionist in Cincinnati . As part of a plan to discredit Stella , he tries to foreclose on Stella 's house , which infuriates her . He then gets arrested and jailed for attacking Myrna Wong , a friend of Stella 's , who rebuffs his advances , throwing him by use of martial arts and destroying the office . John Fiedler as Bobby Taylor , a man on the PTA who lusts after Stella . Overly dominated by his wife , Holly . Although open to allow Stella to speak , he is not above having a fellow PTA board member kidnapped to commit election fraud . He was arrested for indecent exposure after being set up by Stella . Ron Masak as Herbie Maddox , the head bartender at Kelly 's bar , and Stella 's ally against Flora and her friends . He has a crush on Stella , and is one of her friends . Clint Howard as Corley , a young delivery person , who works for the local florist shop , who brings Stella flowers from Will . Susan Swift as Dee Johnson , Stella 's 14 - year - old daughter , who is threatened with expulsion from school if her mother does n't conform to the way the PTA wants her to behave . She has a hand in assisting her mother and Alice with one of their revenge schemes . She loses her braces , gets a makeover , and ends up with a boyfriend , Carlyle . Pat Paulsen as Otis Harper , the descendant of the founding family of Harper Valley . A PTA member who is allied with Flora and her friends . Has a tendency to drink all the time , which annoys his wife , Bertha . He had a stampede of elephants , painted pink , stormed through his house , destroying it . Amzie Strickland as Shirley Thompson , a member of the PTA board who tends to drink a lot of gin . Though she is allied with Flora and her friends , she is one of four PTA members ( Will , Skeeter and Holly Taylor are the others ) who are spared any retribution from Stella . She was seen as the most sympathetic of the board , as she tended to be more manipulated as opposed to being manipulative . It was never revealed whether or not she retained her seat on the PTA , but it is presumed that she became more sympathetic towards Stella as she was seen waving goodbye at Olive as she was being arrested . Bob Hastings as Skeeter Duggan , a member of the PTA board , and the town 's Notary public who is sympathetic to Stella . He is kidnapped by Dutch and Tex at Flora Simpson Reilly 's orders . Audrey Christie as Flora Simpson - Reilly , The movie 's main antagonist . She is the pompous and extremely snobbish president of the PTA board who thoroughly despises Stella ( who defeats her for president at the end of the movie ) and all she stands for . So desperate is she to retain her power and prestige in the community that she has a fellow member of the PTA ( Skeeter Duggan ) kidnapped so she and her allies can commit election fraud . She is embarrassed when her permanent set is destroyed by a hair depilatory at the beauty salon . Molly Dodd as Olive Glover , the recording secretary of the PTA who is a gambling addict . Clearly the most corrupt and most vindictive member of the PTA board , she is eventually arrested for embezzlement after having a load of hay and horse manure dumped on top of her . She is the one who initially angers Stella by sending her the letter denouncing her , and tries to get Dee 's friend , Mavis , arrested for embezzlement . Fay DeWitt as Willamae Jones , a snide biology teacher at the junior high school who is also a member of the PTA . She makes no secret of the fact that she favors the Reilly twins , due to the connections between her and Flora . Notorious for being a nymphomaniac ( though she hides it under a prim - and - proper façade ) , she is caught in flagrante delicto with Barney Crunk , the town 's mailman , on film . DeVera Marcus as Holly Taylor , Bobby 's obviously overbearing wife , who tends to drink a lot when Bobby is out of town , and does n't approve of Bobby 's lusting , or of anything else he does ( this is different from the character in the song , who `` seems to use a lot of ice whenever ( her husband ) 's away '' , implying adultery with the iceman ) . However , despite her ties to Flora and her scheming friends , she , like Shirley Thompson , is spared any retribution from Stella , due to Stella refusing to date Bobby . Louise Foley as Mavis , Dee Johnson 's best friend , who is nearly framed for a crime she did not commit by the vindictive Olive Glover . Brian Cook as Carlyle , a popular track star who chooses Dee , after she gets a makeover , and angers Bettina Reilly in the process . Laura Teige as Bettina Reilly , One of Flora 's twin granddaughters , who wants to be with Carlyle , and is jealous when he chooses Dee over her . She tends to be as snotty as her grandmother . Jan Teige as Edwina Reilly , Bettina 's twin sister , who is just as snotty as her . Irene Yah - Ling Sun as Myrna Wong , a friend of Stella 's who is a judo expert ; she is assaulted by Kirby Baker ( after leading him on ; she destroys him and his office in retaliation ) and has him sent to jail for assault . Tobias Anderson as `` Cranky '' Barney Crunk , the town 's nosy mailman who had the bad habit of peeking in everyone 's mail . He is caught in flagrante delicto on film with Willamae Jones . Royce D. Applegate as Dutch , one of two kidnappers who abduct Skeeter Duggan at Flora Simpson - Reilly 's orders . J.J. Barry as Tex , the mastermind of the kidnap plot to get Skeeter Duggan out of the way so Flora Simpson - Reilly and her friends could commit election fraud . He and his accomplice Dutch are arrested and the two of them rat out Flora and her friends on their scheme . Pitt Herbert as Henry Reilly , Flora 's overly dominated husband and grandfather to Edwina and Bettina . Henry is wealthy , but is very weak - willed and is known for acquiescing to his wife 's schemes . Arlen Stuart as Bertha Harper , wife of Otis Harper , who gets annoyed by her husband 's continual drinking . She is terrified when her house is destroyed by pink - painted elephants .
  Fact 0.0: Barbara Eden plays Stella Johnson.
  Fact 0.1: Stella Johnson is a widowed cosmetics salesperson.
  Fact 0.2: Stella Johnson has a young daughter.
  Fact 0.3: Stella Johnson's joie de vivre angers most of the powerful people in town and on the Parent Teachers Association board.
  Fact 0.4: Stella Johnson's principles anger most of the powerful people in town and on the Parent Teachers Association board.
  Fact 0.5: Stella Johnson's individuality angers most of the powerful people in town and on the Parent Teachers Association board.
  Fact 0.6: Stella Johnson inevitably becomes PTA president after embarrassing Flora, the snobby former President.
  Fact 0.7: Stella Johnson inevitably becomes PTA president after embarrassing Flora's equally snobby friends.
  Fact 0.8: Ronny Cox plays Willis 'Will' Newton.
  Fact 0.9: Will Newton is a wealthy member of the PTA.
  Fact 0.10: Will Newton does not agree with Flora.
  Fact 0.11: Will Newton does not agree with Flora's snobby allies.
  Fact 0.12: Will Newton is in love with Stella.
  Fact 0.13: Will Newton and Stella get married at the end of the movie.
  Fact 0.14: Nanette Fabray plays Alice Finley.
  Fact 0.15: Alice Finley is Stella's best friend.
  Fact 0.16: Alice Finley is Stella's co-conspirator against the snobs of Harper Valley.
  Fact 0.17: Alice Finley owns the La Moderne Beauty Shop.
  Fact 0.18: Alice Finley operates the La Moderne Beauty Shop.
  Fact 0.19: Louis Nye plays Kirby Baker.
  Fact 0.20: Kirby Baker is a local real estate agent.
  Fact 0.21: Kirby Baker is allied with Flora and her friends.
  Fact 0.22: Kirby Baker is notorious for being a lecherous playboy.
  Fact 0.23: Kirby Baker is infamous for impregnating his secretaries.
  Fact 0.24: Gladys Wilmot is one of Kirby Baker's secretaries.
  Fact 0.25: Kirby Baker impregnated Gladys Wilmot.
  Fact 0.26: Gladys Wilmot left town to see an abortionist in Cincinnati.
  Fact 0.27: As part of a plan to discredit Stella, Kirby Baker tries to foreclose on Stella's house.
  Fact 0.28: Stella is infuriated by Kirby Baker's attempt to foreclose on her house.
  Fact 0.29: Kirby Baker gets arrested and jailed for attacking Myrna Wong.
  Fact 0.30: Myrna Wong is a friend of Stella's.
  Fact 0.31: Myrna Wong rebuffs Kirby Baker's advances.
  Fact 0.32: Myrna Wong throws Kirby Baker by use of martial arts.
  Fact 0.33: Myrna Wong destroys Kirby Baker's office.
  Fact 0.34: John Fiedler plays Bobby Taylor.
  Fact 0.35: Bobby Taylor is a man on the PTA.
  Fact 0.36: Bobby Taylor lusts after Stella.
  Fact 0.37: Bobby Taylor is overly dominated by his wife, Holly.
  Fact 0.38: Bobby Taylor is open to allowing Stella to speak.
  Fact 0.39: Bobby Taylor is not above having a fellow PTA board member kidnapped to commit election fraud.
  Fact 0.40: Bobby Taylor was arrested for indecent exposure.
  Fact 0.41: Bobby Taylor was set up by Stella.
  Fact 0.42: Ron Masak plays Herbie Maddox.
  Fact 0.43: Herbie Maddox is the head bartender at Kelly's bar.
  Fact 0.44: Herbie Maddox is Stella's ally against Flora and her friends.
  Fact 0.45: Herbie Maddox has a crush on Stella.
  Fact 0.46: Herbie Maddox is one of Stella's friends.
  Fact 0.47: Clint Howard plays Corley.
  Fact 0.48: Corley is a young delivery person.
  Fact 0.49: Corley works for the local florist shop.
  Fact 0.50: Corley brings Stella flowers from Will.
  Fact 0.51: Susan Swift plays Dee Johnson.
  Fact 0.52: Dee Johnson is Stella's 14-year-old daughter.
  Fact 0.53: Dee Johnson is threatened with expulsion from school if her mother does not conform to the way the PTA wants her to behave.
  Fact 0.54: Dee Johnson has a hand in assisting her mother and Alice with one of their revenge schemes.
  Fact 0.55: Dee Johnson loses her braces.
  Fact 0.56: Dee Johnson gets a makeover.
  Fact 0.57: Dee Johnson ends up with a boyfriend, Carlyle.
  Fact 0.58: Pat Paulsen plays Otis Harper.
  Fact 0.59: Otis Harper is the descendant of the founding family of Harper Valley.
  Fact 0.60: Otis Harper is a PTA member allied with Flora and her friends.
  Fact 0.61: Otis Harper has a tendency to drink all the time.
  Fact 0.62: Otis Harper's drinking annoys his wife, Bertha.
  Fact 0.63: Otis Harper had a stampede of pink-painted elephants storm through his house.
  Fact 0.64: The stampede of pink-painted elephants destroyed Otis Harper's house.
  Fact 0.65: Amzie Strickland plays Shirley Thompson.
  Fact 0.66: Shirley Thompson is a member of the PTA board.
  Fact 0.67: Shirley Thompson tends to drink a lot of gin.
  Fact 0.68: Shirley Thompson is allied with Flora and her friends.
  Fact 0.69: Shirley Thompson is one of four PTA members spared any retribution from Stella.
  Fact 0.70: The other three PTA members spared any retribution from Stella are Will, Skeeter and Holly Taylor.
  Fact 0.71: Shirley Thompson was seen as the most sympathetic member of the board.
  Fact 0.72: Shirley Thompson tended to be more manipulated than manipulative.
  Fact 0.73: It was never revealed whether Shirley Thompson retained her seat on the PTA.
  Fact 0.74: It is presumed that Shirley Thompson became more sympathetic towards Stella.
  Fact 0.75: Shirley Thompson was seen waving goodbye at Olive as Olive was being arrested.
  Fact 0.76: Bob Hastings plays Skeeter Duggan.
  Fact 0.77: Skeeter Duggan is a member of the PTA board.
  Fact 0.78: Skeeter Duggan is the town's Notary public.
  Fact 0.79: Skeeter Duggan is sympathetic to Stella.
  Fact 0.80: Dutch and Tex kidnap Skeeter Duggan at Flora Simpson-Reilly's orders.
  Fact 0.81: Audrey Christie plays Flora Simpson-Reilly.
  Fact 0.82: Flora Simpson-Reilly is the movie's main antagonist.
  Fact 0.83: Flora Simpson-Reilly is the pompous and extremely snobbish president of the PTA board.
  Fact 0.84: Flora Simpson-Reilly thoroughly despises Stella.
  Fact 0.85: Stella defeats Flora Simpson-Reilly for president at the end of the movie.
  Fact 0.86: Flora Simpson-Reilly despises all that Stella stands for.
  Fact 0.87: Flora Simpson-Reilly is desperate to retain her power and prestige in the community.
  Fact 0.88: Flora Simpson-Reilly has a fellow PTA member, Skeeter Duggan, kidnapped so she and her allies can commit election fraud.
  Fact 0.89: Flora Simpson-Reilly is embarrassed when her permanent set is destroyed by a hair depilatory at the beauty salon.
  Fact 0.90: Molly Dodd plays Olive Glover.
  Fact 0.91: Olive Glover is the recording secretary of the PTA.
  Fact 0.92: Olive Glover is a gambling addict.
  Fact 0.93: Olive Glover is clearly the most corrupt and most vindictive member of the PTA board.
  Fact 0.94: Olive Glover is eventually arrested for embezzlement after having a load of hay and horse manure dumped on top of her.
  Fact 0.95: Olive Glover initially angers Stella by sending her the letter denouncing her.
  Fact 0.96: Olive Glover tries to get Dee's friend, Mavis, arrested for embezzlement.
  Fact 0.97: Fay DeWitt plays Willamae Jones.
  Fact 0.98: Willamae Jones is a snide biology teacher at the junior high school.
  Fact 0.99: Willamae Jones is a member of the PTA.
  Fact 0.100: Willamae Jones favors the Reilly twins.
  Fact 0.101: Willamae Jones favors the Reilly twins due to the connections between her and Flora.
  Fact 0.102: Willamae Jones is notorious for being a nymphomaniac.
  Fact 0.103: Willamae Jones hides being a nymphomaniac under a prim-and-proper facade.
  Fact 0.104: Willamae Jones is caught in flagrante delicto with Barney Crunk, the town's mailman, on film.
  Fact 0.105: DeVera Marcus plays Holly Taylor.
  Fact 0.106: Holly Taylor is Bobby's obviously overbearing wife.
  Fact 0.107: Holly Taylor tends to drink a lot when Bobby is out of town.
  Fact 0.108: Holly Taylor does not approve of Bobby's lusting.
  Fact 0.109: Holly Taylor does not approve of anything else Bobby does.
  Fact 0.110: The Holly Taylor character differs from the character in the song, who seems to use a lot of ice whenever her husband is away, implying adultery with the iceman.
  Fact 0.111: Holly Taylor has ties to Flora and her scheming friends.
  Fact 0.112: Holly Taylor, like Shirley Thompson, is spared any retribution from Stella.
  Fact 0.113: Stella spares Holly Taylor due to Stella refusing to date Bobby.
  Fact 0.114: Louise Foley plays Mavis.
  Fact 0.115: Mavis is Dee Johnson's best friend.
  Fact 0.116: Mavis is nearly framed for a crime she did not commit by the vindictive Olive Glover.
  Fact 0.117: Brian Cook plays Carlyle.
  Fact 0.118: Carlyle is a popular track star.
  Fact 0.119: Carlyle chooses Dee after she gets a makeover.
  Fact 0.120: Carlyle's choice of Dee angers Bettina Reilly in the process.
  Fact 0.121: Laura Teige plays Bettina Reilly.
  Fact 0.122: Bettina Reilly is one of Flora's twin granddaughters.
  Fact 0.123: Bettina Reilly wants to be with Carlyle.
  Fact 0.124: Bettina Reilly is jealous when Carlyle chooses Dee over her.
  Fact 0.125: Bettina Reilly tends to be as snotty as her grandmother.
  Fact 0.126: Jan Teige plays Edwina Reilly.
  Fact 0.127: Edwina Reilly is Bettina's twin sister.
  Fact 0.128: Edwina Reilly is just as snotty as Bettina.
  Fact 0.129: Irene Yah-Ling Sun plays Myrna Wong.
  Fact 0.130: Myrna Wong is a friend of Stella's.
  Fact 0.131: Myrna Wong is a judo expert.
  Fact 0.132: Myrna Wong is assaulted by Kirby Baker after she leads him on.
  Fact 0.133: Myrna Wong destroys Kirby Baker and his office in retaliation.
  Fact 0.134: Myrna Wong has Kirby Baker sent to jail for assault.
  Fact 0.135: Tobias Anderson plays 'Cranky' Barney Crunk.
  Fact 0.136: Barney Crunk is the town's nosy mailman.
  Fact 0.137: Barney Crunk had the bad habit of peeking in everyone's mail.
  Fact 0.138: Barney Crunk is caught in flagrante delicto on film with Willamae Jones.
  Fact 0.139: Royce D. Applegate plays Dutch.
  Fact 0.140: Dutch is one of two kidnappers who abduct Skeeter Duggan at Flora Simpson-Reilly's orders.
  Fact 0.141: J.J. Barry plays Tex.
  Fact 0.142: Tex is the mastermind of the kidnap plot to get Skeeter Duggan out of the way.
  Fact 0.143: The kidnap plot lets Flora Simpson-Reilly and her friends commit election fraud.
  Fact 0.144: Tex is arrested.
  Fact 0.145: Dutch is arrested.
  Fact 0.146: Tex and Dutch rat out Flora and her friends on their scheme.
  Fact 0.147: Pitt Herbert plays Henry Reilly.
  Fact 0.148: Henry Reilly is Flora's overly dominated husband.
  Fact 0.149: Henry Reilly is the grandfather of Edwina and Bettina.
  Fact 0.150: Henry Reilly is wealthy.
  Fact 0.151: Henry Reilly is very weak-willed.
  Fact 0.152: Henry Reilly is known for acquiescing to his wife's schemes.
  Fact 0.153: Arlen Stuart plays Bertha Harper.
  Fact 0.154: Bertha Harper is the wife of Otis Harper.
  Fact 0.155: Bertha Harper gets annoyed by her husband's continual drinking.
  Fact 0.156: Bertha Harper is terrified when her house is destroyed by pink-painted elephants.

Retrieved items:
<item 1>
Harper Valley PTA (film)
She is terrified when her house is destroyed by pink - painted elephants . Production and release ( edit ) Filming ( edit ) Harper Valley PTA was filmed in
twenty - seven days from October 3 , 1977 to December 8 , 1977 . It was filmed on location for one week in the town of Lebanon , Ohio and then continued in Los
Angeles , California . The track meet scene was filmed at Simi Valley High School On October 31 , 1977 , production ceased temporarily when Nanette Fabray was
knocked to the ground by a spooked elephant while shooting a scene involving pink elephants in the Hancock Park neighborhood of Los Angeles , California ; she
suffered a severe concussion , bruises and back sprains and was taken to Cedars - Sinai Medical Center where her condition was reported as serious but stable .
Following her recovery , filming resumed on November 30 . Release ( edit ) Harper Valley PTA opened in six theaters in Lebanon , Cincinnati and Dayton , Ohio on
May 23 , 1978 in accordance with executive producer Phil Borack 's plan to release the film first in smaller markets , where regional success could encourage
exhibitors in bigger cities to book the film .
</item 1>

<item 2>
Harper Valley PTA (film)
He was arrested for indecent exposure after being set up by Stella . Ron Masak as Herbie Maddox , the head bartender at Kelly 's bar , and Stella 's ally
against Flora and her friends . He has a crush on Stella , and is one of her friends . Clint Howard as Corley , a young delivery person , who works for the
local florist shop , who brings Stella flowers from Will . Susan Swift as Dee Johnson , Stella 's 14 - year - old daughter , who is threatened with expulsion
from school if her mother does n't conform to the way the PTA wants her to behave . She has a hand in assisting her mother and Alice with one of their revenge
schemes . She loses her braces , gets a makeover , and ends up with a boyfriend , Carlyle . Pat Paulsen as Otis Harper , the descendant of the founding family
of Harper Valley . A PTA member who is allied with Flora and her friends . Has a tendency to drink all the time , which annoys his wife , Bertha . He had a
stampede of elephants , painted pink , stormed through his house , destroying it .
</item 2>

<item 3>
Harper Valley PTA (film)
The film starred Barbara Eden , Nanette Fabray , Ronny Cox , Louis Nye and Susan Swift , directed by Richard Bennett , and was mainly released to drive - in
theaters throughout the summer of 1978 . The film has a cult following in fans of the original song . The film 's promotional tagline is : `` The song was
scandalous . The movie is hilarious ! '' Contents ( hide ) 1 Plot 2 Cast and characters 3 Production and release 3.1 Filming 3.2 Release 4 Soundtrack 5
Television series 6 References 7 External links Plot ( edit ) Stella Johnson is a beautiful widowed single mother who lives in the town of Harper Valley , Ohio
. She sells cosmetics door - to - door for the fictitious AngelGlo Cosmetics and is not afraid to enjoy life . Her fourteen - year - old daughter , Dee , is a
student at Harper Valley Junior High School . After leaving school , Dee brings her mother a letter from the school 's Parent Teacher Association board , which
is led by the pompous and snobbish Flora Simpson - Reilly .
</item 3>

<item 4>
Harper Valley PTA (film)
The film grossed over $2 million its opening weekend , $5.8 million in its first three weeks and over $16 million after twelve weeks . The film opened in Los
Angeles on August 2 , 1978 and in New York City on January 12 , 1979 . Soundtrack ( edit ) Harper Valley PTA : Original Soundtrack Recording Soundtrack album by
various artists Released June 1978 Recorded Singleton Sound Studios Nashville , Tennessee Genre Soundtrack Length 29 : 50 Label Plantation Records Producer
Shelby S. Singleton , Jr . Singles from Harper Valley PTA `` Harper Valley PTA ( re-issue ) '' Released : July 1978 In June 1978 , a soundtrack album was
released on vinyl , cassette and 8 - track tape by Plantation Records . To promote the film 's release and its soundtrack , the title song by Jeannie C. Riley
was re-issued as a single . The album made its compact disc premiere through Varèse Sarabande on October 27 , 2017 .
</item 4>

<item 5>
Harper Valley PTA (film)
He is caught in flagrante delicto on film with Willamae Jones . Royce D. Applegate as Dutch , one of two kidnappers who abduct Skeeter Duggan at Flora Simpson -
Reilly 's orders . J.J. Barry as Tex , the mastermind of the kidnap plot to get Skeeter Duggan out of the way so Flora Simpson - Reilly and her friends could
commit election fraud . He and his accomplice Dutch are arrested and the two of them rat out Flora and her friends on their scheme . Pitt Herbert as Henry
Reilly , Flora 's overly dominated husband and grandfather to Edwina and Bettina . Henry is wealthy , but is very weak - willed and is known for acquiescing to
his wife 's schemes . Arlen Stuart as Bertha Harper , wife of Otis Harper , who gets annoyed by her husband 's continual drinking . She is terrified when her
house is destroyed by pink - painted elephants .
</item 5>

<item 6>
Harper Valley PTA (film)
Harper Valley PTA ( film ) - wikipedia Harper Valley PTA ( film ) Jump to : navigation , search Harper Valley PTA Theatrical release poster Directed by Richard
Bennett Produced by George Edwards Written by George Edwards Barry Schneider Starring Barbara Eden Nanette Fabray Ronny Cox Louis Nye Susan Swift Pat Paulsen
Music by Nelson Riddle Cinematography Willy Kurant Edited by Michael Economou Distributed by April Fools Productions Release date May 23 , 1978 ( 1978 - 05 - 23
) ( Midwest ) June 2 , 1978 ( 1978 - 06 - 02 ) ( Canada ) August 2 , 1978 ( 1978 - 08 - 02 ) ( Los Angeles ) January 12 , 1979 ( 1979 - 01 - 12 ) ( New York
City ) Running time 93 minutes Country United States Language English Budget $1 million Box office $25 million ( USA ) Harper Valley PTA is a 1978 American
comedy film inspired by the popular 1968 country song `` Harper Valley PTA '' written by Tom T. Hall and performed by country singer Jeannie C. Riley .
</item 6>

<item 7>
Harper Valley PTA (film)
Barbara Eden reprised her role as Stella Johnson for the series which lasted two seasons and a total of 30 episodes were produced . References ( edit ) Jump up
^ St. Petersburg Times Jump up ^ Milwaukee Sentinel : Elephant injures Nanette Fabray Jump up ^ Observer Reporter External links ( edit ) Harper Valley PTA on
IMDb Harper Valley PTA at the TCM Movie Database Harper Valley PTA at AllMovie Harper Valley PTA at the American Film Institute Catalog Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Harper_Valley_PTA_(film)&oldid=818590841 '' Categories : 1978 films English - language films 1970s comedy films
American comedy films American films American independent films Feminist films Films about revenge Films adapted into television programs Films based on songs
Films set in Ohio Films shot in Ohio Films shot in Los Angeles 1978 soundtracks Film soundtracks Hidden categories : Articles with hAudio microformats Album
infoboxes lacking a cover Music infoboxes with deprecated parameters Talk Contents About Wikipedia Add links This page was last edited on 4 January 2018 , at 12
: 58 .
</item 7>

<item 8>
Harper Valley PTA (film)
Side 1 `` Harper Valley PTA '' -- performed by Jeannie C. Riley `` Dee 's Visit '' ( instrumental ) -- performed by Nelson Riddle `` Mr. Harper '' -- performed
by Barbara Eden `` Alice 's Place '' ( instrumental ) -- performed by Nelson Riddle `` High School Confidential '' -- performed by Jerry Lee Lewis `` Willie Mae
'' ( instrumental ) -- performed by Nelson Riddle Side 2 `` Harper Valley PTA '' ( instrumental ) -- performed by Nelson Riddle `` Widow Jones '' -- performed
by Barbara Eden `` Twin Tune '' ( instrumental ) -- performed by Nelson Riddle `` Ballad of a Teenage Queen '' -- performed by Johnny Cash `` Ice Cream Disco ''
( instrumental ) -- performed by Nelson Riddle `` Whatever Happened to Charlie Brown '' -- performed by Rita Remington & Carol Channing `` Ending ( Reprise :
Harper Valley PTA ) '' ( instrumental ) -- performed by Nelson Riddle Television series ( edit ) Main article : Harper Valley PTA ( TV series ) In 1981 , Harper
Valley PTA was made into a television sitcom ( created by Sherwood Schwartz ) which aired on NBC from January 1981 to May 1982 .
</item 8>

<item 9>
Harper Valley PTA (film)
Also shown are Edwina , Bettina 's identical twin sister who is just as snobbish as Bettina , and Dee 's best friend Mavis . Real estate agent Kirby Baker , a
member of the PTA Board , plans to ruin Stella by foreclosing on her house ( which his company owns ) , but is arrested for assaulting Myrna Wong , an Asian -
American martial arts expert helping Stella to set up the lecherous Baker . Now things become more and more desperate . The board finally decide to resort to
criminal means to maintain the power they hold , which is fast slipping away thanks to Stella 's growing popularity . The board members then decide to hire a
couple of kidnappers named Dutch and Tex to have Skeeter abducted so they can commit election fraud . The kidnappers snatch Skeeter as he is taking the trash
out and take him in a nearby abbey , where he is imprisoned and made drunk on wine .
</item 9>

<item 10>
Harper Valley PTA (film)
He had a stampede of elephants , painted pink , stormed through his house , destroying it . Amzie Strickland as Shirley Thompson , a member of the PTA board who
tends to drink a lot of gin . Though she is allied with Flora and her friends , she is one of four PTA members ( Will , Skeeter and Holly Taylor are the others
) who are spared any retribution from Stella . She was seen as the most sympathetic of the board , as she tended to be more manipulated as opposed to being
manipulative . It was never revealed whether or not she retained her seat on the PTA , but it is presumed that she became more sympathetic towards Stella as she
was seen waving goodbye at Olive as she was being arrested . Bob Hastings as Skeeter Duggan , a member of the PTA board , and the town 's Notary public who is
sympathetic to Stella . He is kidnapped by Dutch and Tex at Flora Simpson Reilly 's orders . Audrey Christie as Flora Simpson - Reilly , The movie 's main
antagonist .
</item 10>
=== END ITEM id=dc64e3234bd1b571 ===
