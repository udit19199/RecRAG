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

This file has 7 items.

=== ITEM id=d0ca0dcba3beea0a ===
Gold sentences and facts:
Sentence 0: Lion 's mane jellyfish Scientific classification Kingdom : Animalia Phylum : Cnidaria Class : Scyphozoa Order : Semaeostomeae Family : Cyaneidae Genus : Cyanea Species : C. capillata Binomial name Cyanea capillata ( Linnaeus , 1758 )
  Fact 0.0: The lion's mane jellyfish has the kingdom Animalia.
  Fact 0.1: The lion's mane jellyfish has the phylum Cnidaria.
  Fact 0.2: The lion's mane jellyfish has the class Scyphozoa.
  Fact 0.3: The lion's mane jellyfish has the order Semaeostomeae.
  Fact 0.4: The lion's mane jellyfish has the family Cyaneidae.
  Fact 0.5: The lion's mane jellyfish has the genus Cyanea.
  Fact 0.6: The lion's mane jellyfish has the species C. capillata.
  Fact 0.7: The binomial name of the lion's mane jellyfish is Cyanea capillata.
  Fact 0.8: The binomial name Cyanea capillata was given by Linnaeus in 1758.
Sentence 1: The lion 's mane jellyfish ( Cyanea capillata ) , also known as the giant jellyfish or the hair jelly , is the largest known species of jellyfish . Its range is confined to cold , boreal waters of the Arctic , northern Atlantic , and northern Pacific Oceans . It is common in the English Channel , Irish Sea , North Sea and in western Scandinavian waters south to Kattegat and Øresund . It may also drift in to the south - western part of the Baltic Sea ( where it can not breed due to the low salinity ) . Similar jellyfish -- which may be the same species -- are known to inhabit seas near Australia and New Zealand . The largest recorded specimen found washed up on the shore of Massachusetts Bay in 1870 , had a bell with a diameter of 2.3 metres ( 7 ft 6 in ) and tentacles 37.0 m ( 121.4 ft ) long . Lion 's mane jellyfish have been observed below 42 ° N latitude for some time in the larger bays of the east coast of the United States .
  Fact 1.0: The lion's mane jellyfish (Cyanea capillata) is also known as the giant jellyfish or the hair jelly.
  Fact 1.1: The lion's mane jellyfish is the largest known species of jellyfish.
  Fact 1.2: The range of the lion's mane jellyfish is confined to cold, boreal waters of the Arctic, northern Atlantic, and northern Pacific Oceans.
  Fact 1.3: The lion's mane jellyfish is common in the English Channel, Irish Sea, and North Sea.
  Fact 1.4: The lion's mane jellyfish is common in western Scandinavian waters south to Kattegat and Oresund.
  Fact 1.5: The lion's mane jellyfish may drift into the south-western part of the Baltic Sea.
  Fact 1.6: The lion's mane jellyfish cannot breed in the Baltic Sea due to the low salinity.
  Fact 1.7: Similar jellyfish, which may be the same species, are known to inhabit seas near Australia and New Zealand.
  Fact 1.8: The largest recorded specimen washed up on the shore of Massachusetts Bay in 1870.
  Fact 1.9: The largest recorded specimen had a bell with a diameter of 2.3 metres (7 ft 6 in).
  Fact 1.10: The largest recorded specimen had tentacles 37.0 m (121.4 ft) long.
  Fact 1.11: The lion's mane jellyfish has been observed below 42 degrees N latitude for some time.
  Fact 1.12: The lion's mane jellyfish has been observed in the larger bays of the east coast of the United States.

Retrieved items:
<item 1>
Lion's mane jellyfish
Considering the size of the species , it is possible that this mass incident was caused by a single specimen . In popular culture ( edit ) The lion 's mane
jellyfish appears in the Sherlock Holmes short story `` The Adventure of the Lion 's Mane '' published in The Case - Book of Sherlock Holmes . Holmes discovers
at the end of the story that the true killer of a professor who died shortly after going swimming ( shouting `` the lion 's mane '' before he succumbed ) was
actually this jellyfish . Suspicion was originally laid upon the professor 's rival in love , until the latter was similarly attacked ( he survived , although
badly stung ) . In the context of the story , it is only because the school professor has a weak heart that he succumbs , as is confirmed by the survival of the
second victim . A photograph widely distributed on the internet appears to show an anomalously large lion 's mane dwarfing a nearby diver by several times . The
photo was subsequently shown to be a hoax . On the popular television program QI the show claimed that the longest animal in the world was the lion 's mane
jellyfish .
</item 1>

<item 2>
Lion's mane jellyfish
The largest recorded specimen found washed up on the shore of Massachusetts Bay in 1870 , had a bell with a diameter of 2.3 metres ( 7 ft 6 in ) and tentacles
37.0 m ( 121.4 ft ) long . Lion 's mane jellyfish have been observed below 42 ° N latitude for some time in the larger bays of the east coast of the United
States . The lion 's mane jellyfish uses its stinging tentacles to capture , pull in and eat prey such as fish , sea creatures and smaller jellyfish . Contents
( hide ) 1 Taxonomy 2 Description 3 Ecology 4 Behavior and reproduction 5 Sting and human contact 6 In popular culture 7 Gallery 8 References 8.1 Other sources
9 External links Taxonomy ( edit ) Cyanea sp . The taxonomy of the Cyanea species is not fully agreed upon ; some zoologists have suggested that all species
within the genus should be treated as one .
</item 2>

<item 3>
Lion's mane jellyfish
Polar Biol . doi : 10.1007 / s00300 - 015 - 1707 - y . Jump up ^ `` Lion 's Mane Jellyfish '' . Scubatravel.co.uk . Retrieved 2010 - 09 - 03 . Jump up ^
https://web.archive.org/web/20110108024017/http://www.typesofjellyfish.net/lions_mane_jellyfish/lions_mane_jellyfish.html Jump up ^ Kozloff , Eugene ( 2003 ) .
Seashore Life of the Northern Pacific Coast . University of Washington Press . pp. 54 , 56 ( 1983 edition ) . ISBN 0295960841 . Jump up ^ Point 6 at ^ Jump up
to : `` Lions Mane Jellyfish '' . Types Of Jellyfish . Archived from the original on 2011 - 01 - 08 . Retrieved 2010 - 09 - 03 .
</item 3>

<item 4>
Lion's mane jellyfish
Lion 's mane jellyfish - wikipedia Lion 's mane jellyfish Jump to : navigation , search Lion 's mane jellyfish Scientific classification Kingdom : Animalia
Phylum : Cnidaria Class : Scyphozoa Order : Semaeostomeae Family : Cyaneidae Genus : Cyanea Species : C. capillata Binomial name Cyanea capillata ( Linnaeus ,
1758 ) The lion 's mane jellyfish ( Cyanea capillata ) , also known as the giant jellyfish or the hair jelly , is the largest known species of jellyfish . Its
range is confined to cold , boreal waters of the Arctic , northern Atlantic , and northern Pacific Oceans . It is common in the English Channel , Irish Sea ,
North Sea and in western Scandinavian waters south to Kattegat and Øresund . It may also drift in to the south - western part of the Baltic Sea ( where it can
not breed due to the low salinity ) . Similar jellyfish -- which may be the same species -- are known to inhabit seas near Australia and New Zealand .
</item 4>

<item 5>
Lion's mane jellyfish
This was later corrected -- in 1864 , a bootlace worm ( Lineus longissimus ) was found washed up on a Scottish shore that was 55 m ( 180 ft ) long . ( That
claim has its critics : because bootlace worms can easily stretch to several times their natural length , it is possible the worm did not actually grow to be
that length . ) Gallery ( edit ) A lion 's mane jelly capturing a sea gooseberry Lion 's mane jellyfish with full threads visible Lion 's mane jellyfish
contracted Lion 's mane jellyfish in star formation Lion 's mane jellyfish open , showing underside Life - sized model in the Smithsonian Museum of Natural
History , Washington DC , US Small , dead lion 's mane jelly washing up on the beach Blue lion 's mane jellyfish in Askeröfjorden outside Stenungsund , Sweden .
References ( edit ) Jump up ^ Hair Jelly ( Cyanea capillata ) . Australian Venom Research Unit . ^ Jump up to : `` Lion 's Mane Jellyfish -- Reference Library
'' . redOrbit . Retrieved 2010 - 09 - 03 .
</item 5>

<item 6>
Lion's mane jellyfish
Two distinct taxa , however , occur together in at least the eastern North Atlantic , with the blue jellyfish ( Cyanea lamarckii Péron & Lesueur , 1810 )
differing in color ( blue , not red ) and smaller size ( 10 -- 20 cm diameter , rarely 35 cm ) . Populations in the western Pacific around Japan are sometimes
distinguished as Cyanea nozakii , or as a subspecies , C. c. nozakii . In 2015 , Russian researchers announced a possible sister species , Cyanea tzetlinii
found in the White Sea , but this has not yet been recognized by other authoritative databases such as WoRMS or ITIS . Description ( edit ) Lion 's mane
jellyfish ( Cyanea capillata ) are named for their showy , trailing tentacles reminiscent of a lion 's mane . They can vary greatly in size : although capable
of attaining a bell diameter of over 2 metres ( 6 ft 7 in ) , those found in lower latitudes are much smaller than their far northern counterparts , with a bell
about 50 centimetres ( 20 in ) in diameter .
</item 6>

<item 7>
Lion's mane jellyfish
Size seems to dictate coloration : larger specimens are a vivid crimson to dark purple while smaller specimens grade to a lighter orange or tan , occasionally
colorless . The bell of the lion 's mane jellyfish is scalloped into eight lobes ( lappets ) , each lobe containing from 70 to 150 tentacles , arranged in four
fairly distinct rows . Along the bell margin is a balance organ at each of the eight indentations between the lobes -- the rhopalium -- which helps the
jellyfish orient itself . From the central mouth extend broad frilly oral arms with many stinging cells . Closer to its mouth , its total number of tentacles
are around 1200 . The long , thin tentacles which emanate from the bell 's subumbrella have been characterised as `` extremely sticky '' ; they also have
stinging cells . The tentacles of larger specimens may trail as long as 30 m ( 100 ft ) or more , with the tentacles of the longest known specimen measured at
37 m ( 120 ft ) in length .
</item 7>

<item 8>
Lion's mane jellyfish
redOrbit . Retrieved 2010 - 09 - 03 . Jump up ^ `` Lion 's Mane Jellyfish : Discovery Channel '' . Dsc.discovery.com. 2013 - 05 - 23 . Retrieved 2013 - 09 - 08
. Jump up ^ Glafira D. Kolbasova ; Arthur O. Zalevsky ; Azamat R. Gafurov ; Philipp O. Gusev ; Margarita A. Ezhova ; Anna A. Zheludkevich ; Olga P. Konovalova ;
Ksenia N. Kosobokova ; Nikita U. Kotlov ; Natalia O. Lanina ; Anna S. Lapashina ; Dmitry O. Medvedev ; Katerina S. Nosikova ; Ekaterina O. Nuzhdina ; Georgii A.
Bazykin ; Tatyana V. Neretina ( 2015 ) . `` A new species of Cyanea jellyfish sympatric to C. capillata in the White Sea '' . Polar Biol .
</item 8>

<item 9>
Lion's mane jellyfish
wikipedia.org/w/index.php?title=Lion%27s_mane_jellyfish&oldid=834313563 '' Categories : Semaeostomeae Animals described in 1758 Hidden categories : Articles
with ' species ' microformats All articles with unsourced statements Articles with unsourced statements from February 2012 Talk Contents About Wikipedia
Wikispecies Azərbaycanca Català Cebuano Čeština Dansk Deutsch Español Esperanto فارسی Français Frysk Gaeilge 한국어 Bahasa Indonesia Íslenska Italiano עברית
Lietuvių Magyar Nederlands Norsk Norsk nynorsk پنجابی Polski Português Română Русский Simple English Suomi Svenska Türkçe Tiếng Việt Winaray 中文 26 more Edit
links This page was last edited on 5 April 2018 , at 01 : 30 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms
may apply . By using this site , you agree to the Terms of Use and Privacy Policy .
</item 9>

<item 10>
Lion's mane jellyfish
Behavior and reproduction ( edit ) Main article : Jellyfish § Life cycle Lion 's mane jellyfish swimming , side view Lion 's mane jellyfish remain mostly very
near the surface , at no more than 20 m depth . Their slow pulsations drive them forwards weakly , so they depend on ocean currents to travel great distances .
The jellyfish are most often spotted during the late summer and autumn , when they have grown to a large size and the currents begin to sweep them to shore .
Like other jellyfish , lion 's manes are capable of both sexual reproduction in the medusa stage and asexual reproduction in the polyp stage . Lion 's mane
jellyfish have four different stages in their year - long lifespan : a larval stage , a polyp stage , an ephyrae stage , and the medusa stage . The female
jellyfish carries its fertilized eggs in its tentacle , where the eggs grow into larva . When the larva are old enough , the female deposits them on a hard
surface , where the larva soon grow into polyps . The polyps begin to reproduce asexually , creating stacks of small creatures called ephyrae .
</item 10>
=== END ITEM id=d0ca0dcba3beea0a ===

=== ITEM id=136401815602423c ===
Gold sentences and facts:
Sentence 0: Islami Somaz Bank Limited Al - Arafah Islami Bank Limited EXIM Bank Limited First Security Islami Bank Limited ICB Islamic Bank Limited Islami Bank Bangladesh Limited Shahjalal Islami Bank Limited Social Islami Bank Limited Union Bank Limited
  Fact 0.0: Islami Somaz Bank Limited is listed as a bank.
  Fact 0.1: Al - Arafah Islami Bank Limited is listed as a bank.
  Fact 0.2: EXIM Bank Limited is listed as a bank.
  Fact 0.3: First Security Islami Bank Limited is listed as a bank.
  Fact 0.4: ICB Islamic Bank Limited is listed as a bank.
  Fact 0.5: Islami Bank Bangladesh Limited is listed as a bank.
  Fact 0.6: Shahjalal Islami Bank Limited is listed as a bank.
  Fact 0.7: Social Islami Bank Limited is listed as a bank.
  Fact 0.8: Union Bank Limited is listed as a bank.

Retrieved items:
<item 1>
Economy of India
The total bank deposits increased from ₹ 59.1 billion ( US $910 million ) in 1970 -- 71 to ₹ 38,309.22 billion ( US $590 billion ) in 2008 -- 09 . Despite an
increase of rural branches -- from 1,860 or 22 % of the total in 1969 to 30,590 or 42 % in 2007 -- only 32,270 of 500,000 villages are served by a scheduled
bank . India 's gross domestic savings in 2006 -- 07 as a percentage of GDP stood at a high 32.8 % . More than half of personal savings are invested in physical
assets such as land , houses , cattle , and gold . The government - owned public - sector banks hold over 75 % of total assets of the banking industry , with
the private and foreign banks holding 18.2 % and 6.5 % respectively . Since liberalisation , the government has approved significant banking reforms .
</item 1>

<item 2>
List of banks in Bangladesh
National Finance Limited National Housing Finance and Investments Limited People 's Leasing and Financial Services Limited Phoenix Finance and Investments
Limited Premier Leasing & Finance Limited Prime Finance & Investment Limited Reliance Finance Limited Saudi - Bangladesh Industrial & Agricultural Investment
Company Limited SEAF Bangladesh Venture Limited The UAE - Bangladesh Investment Company Limited Union Capital Limited United Finance Limited Uttara Finance and
Investments Limited Specialized Financial Institutions ( Semi Formal Sector ) ( edit ) House Building Financial Corporation ( HBFC ) Palli Karma Sahayak
Foundation ( PKSF ) References ( edit ) ^ Jump up to : `` Financial System '' . Central Bank of Bangladesh . Retrieved 9 August 2016 . Jump up ^ `` Relevant
links '' . Central Bank of Bangladesh . Retrieved 7 February 2018 .
</item 2>

<item 3>
List of banks in Bangladesh
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 3>

<item 4>
Toyota
2000s ( edit ) In 2001 , Toyota 's Toyo Trust and Banking merged with two other banks to form UFJ Bank , which was accused of corruption by Japan 's government
for making bad loans to alleged Yakuza crime syndicates with executives accused of blocking Financial Service Agency inspections . The UFJ was listed among
Fortune Magazine 's largest money - losing corporations in the world , with Toyota 's chairman serving as a director . At the time , the UFJ was one of the
largest shareholders of Toyota . As a result of Japan 's banking crisis , UFJ merged with the Bank of Tokyo - Mitsubishi to become the Mitsubishi UFJ Financial
Group . In 2002 , Toyota managed to enter a Formula One works team and establish joint ventures with French motoring companies Citroën and Peugeot a year after
Toyota started producing cars in France . Toyota ranked eighth on Forbes 2000 list of the world 's leading companies for the year 2005 but slid to 55 for 2011 .
The company was number one in global automobile sales for the first quarter of 2008 .
</item 4>

<item 5>
List of banks in Bangladesh
Rupali Bank Ltd . Bangladesh Krishi Bank Bangladesh Development Bank Limited . Basic Bank Limited Private Commercial Banks ( edit ) AB Bank Limited Bangladesh
Commerce Bank Limited Bank Asia Limited Bengal Bank Limited BRAC Bank Limited City Bank Limited Dhaka Bank Limited Dutch - Bangla Bank Limited Eastern Bank
Limited IFIC Bank Limited Jamuna Bank Limited Meghna Bank Limited Mercantile Bank Limited Midland Bank Modhumoti Bank Limited Mutual Trust Bank Limited National
Bank Limited National Credit & Commerce Bank Limited NRB Bank Limited NRB Commercial Bank Limited One Bank Limited Premier Bank Limited Prime Bank Limited
Pubali Bank Limited South Bangla Agriculture & Commerce Bank Limited Southeast Bank Limited Standard Bank Limited The Farmers Bank Limited Trust Bank Limited
United Commercial Bank Limited Uttara Bank Limited Shimanto Bank Ltd Islamic Commercial Private Banks ( edit ) Islami Somaz Bank Limited Al - Arafah Islami Bank
Limited EXIM Bank Limited First Security Islami Bank Limited ICB Islamic Bank Limited Islami Bank Bangladesh Limited Shahjalal Islami Bank Limited Social Islami
Bank Limited Union Bank Limited Foreign Commercial Banks ( edit ) Bank Al - Falah Limited Citibank N.
</item 5>

<item 6>
Economy of India
The Indian money market is classified into the organised sector , comprising private , public and foreign - owned commercial banks and cooperative banks ,
together known as ' scheduled banks ' ; and the unorganised sector , which includes individual or family - owned indigenous bankers or money lenders and
non-banking financial companies . The unorganised sector and microcredit are preferred over traditional banks in rural and sub-urban areas , especially for
non-productive purposes such as short - term loans for ceremonies . Prime Minister Indira Gandhi nationalised 14 banks in 1969 , followed by six others in 1980
, and made it mandatory for banks to provide 40 % of their net credit to priority sectors including agriculture , small - scale industry , retail trade and
small business , to ensure that the banks fulfilled their social and developmental goals . Since then , the number of bank branches has increased from 8,260 in
1969 to 72,170 in 2007 and the population covered by a branch decreased from 63,800 to 15,000 during the same period .
</item 6>

<item 7>
List of banks in Bangladesh
Central Bank of Bangladesh . Retrieved 7 February 2018 . ( hide ) List of banks in Asia Sovereign states Afghanistan Armenia Azerbaijan Bahrain Bangladesh
Bhutan Brunei Cambodia China Cyprus East Timor ( Timor - Leste ) Egypt Georgia India Indonesia Iran Iraq Israel Japan Jordan Kazakhstan North Korea South Korea
Kuwait Kyrgyzstan Laos Lebanon Malaysia Maldives Mongolia Myanmar Nepal Oman Pakistan Philippines Qatar Russia Saudi Arabia Singapore Sri Lanka Syria Tajikistan
Thailand Turkey Turkmenistan United Arab Emirates Uzbekistan Vietnam Yemen States with limited recognition Abkhazia Artsakh Northern Cyprus Palestine South
Ossetia Taiwan Dependencies and other territories British Indian Ocean Territory Christmas Island Cocos ( Keeling ) Islands Hong Kong Macau Retrieved from ``
https://en.wikipedia.org/w/index.php?title=List_of_banks_in_Bangladesh&oldid=832850714 '' Categories : Lists of companies of Bangladesh Lists of banks by
country Banks of Bangladesh Lists of banks Talk Contents About Wikipedia বাংলা Edit links This page was last edited on 28 March 2018 , at 10 : 02 . Text is
available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
</item 7>

<item 8>
Agriculture in India
Reserve Bank of India Governors India Government Mint Financial services Banking Banks Insurance Bombay Stock Exchange National Stock Exchange Multi Commodity
Exchange Bullion Black money Stock exchanges History Economic development Economic liberalisation Licence Raj Green revolution Government initiatives Numbering
system People Billionaires Businesspeople Demography Income Poverty Labour law Pensions EPFO NPS PPF States Andhra Pradesh Assam Bihar Delhi Goa Gujarat Haryana
Himachal Pradesh Jammu and Kashmir Karnataka Kerala Madhya Pradesh Maharashtra Mizoram Odisha Punjab Rajasthan Tamil Nadu Telangana Uttarakhand Uttar Pradesh
West Bengal Sectors Agriculture Livestock Fishing Automotive Defence Construction Education Energy Nuclear Solar Wind Entertainment Forestry Gambling Healthcare
Information technology Media Cinema Television Printing Mining Coal Iron and Steel Pharmaceuticals Retail Science and technology Biotechnology
Telecommunications Textiles Tourism Transport Aviation Civil Ports Rail Utilities Electricity Water Commons Wikiquotes Agriculture in Asia Sovereign states
Afghanistan Armenia Azerbaijan Bahrain Bangladesh Bhutan Brunei Cambodia China Cyprus East Timor ( Timor - Leste ) Egypt Georgia India Indonesia Iran Iraq
Israel Japan Jordan Kazakhstan North Korea South Korea Kuwait Kyrgyzstan Laos Lebanon Malaysia Maldives Mongolia Myanmar Nepal Oman Pakistan Philippines Qatar
Russia Saudi Arabia Singapore Sri Lanka Syria Tajikistan Thailand Turkey Turkmenistan United Arab Emirates Uzbekistan Vietnam Yemen States with limited
recognition Abkhazia
</item 8>

<item 9>
List of banks in Bangladesh
List of banks in Bangladesh - wikipedia List of banks in Bangladesh Jump to : navigation , search The banking sector in Bangladesh consists of several types of
institution : Contents ( hide ) 1 Central bank 2 Scheduled Banks 3 State Owned Banks 4 Private Commercial Banks 5 Islamic Commercial Private Banks 6 Foreign
Commercial Banks 7 Non-Scheduled Banks 8 Non Bank Financial Institutions 9 Specialized Financial Institutions ( Semi Formal Sector ) 10 References Central Bank
( edit ) Bangladesh Bank is the central bank of Bangladesh and the chief regulatory authority in the banking sector . Pursuant to Bangladesh Bank Order , 1972
the Government of Bangladesh reorganized the Dhaka Branch of the State Bank of Pakistan as the central bank of the country , and named it Bangladesh Bank with
retrospective effect from 16 December 1971 . Scheduled banks ( edit ) Secheduled Banks are licensed under the Bank Company Act , 1991 ( Amended up to 2013 ) .
State Owned Banks ( edit ) Sonali Bank Ltd . Janata Bank Ltd . Agrani Bank Ltd . Rupali Bank Ltd . Bangladesh Krishi Bank Bangladesh Development Bank Limited .
</item 9>

<item 10>
List of banks in Bangladesh
A Commercial Bank of Ceylon PLC Habib Bank Limited National Bank of Pakistan Standard Chartered Bank State Bank of India Punjab National Bank Woori Bank HSBC
Non-scheduled banks ( edit ) Non-Scheduled Banks are licensed only for specific functions and objectives , and do not offer the same range of services as
scheduled banks . Ansar VDP Unnayan Bank Karmashangosthan Bank Probashi Kollyan Bank Grameen Bank Jubilee Bank Palli Sanchay Bank Non Bank Financial
Institutions ( edit ) Investment Corporation of Bangladesh Agrani SME Financing Company Limited Bangladesh Finance and Investment Company Limited Bangladesh
Industrial Finance Company Limited Bangladesh Infrastructure Finance Fund Limited Bay Leasing and Investment Limited CAPM Venture Capital and Finance Limited
Delta Brac Housing Finance Corporation Limited Far - east Finance & Investment Limited FAS Finance & Investment Limited First Finance Limited GSP Finance
Company ( Bangladesh ) Limited Hajj Finance Company Limited IDLC Finance Limited Industrial and Infrastructure Development Finance Company Limited IPDC Finance
Limited Infrastructure Development Company Limited International Leasing and Financial Services Limited Islamic Finance and Investment Limited LankaBangla
Finance Limited Meridian Finance and Investment Limited MIDAS Financing Limited National Finance Limited National Housing Finance and Investments Limited People
's Leasing and Financial Services Limited Phoenix Finance
</item 10>
=== END ITEM id=136401815602423c ===

=== ITEM id=e54c2d561047a874 ===
Gold sentences and facts:
Sentence 0: Mac OS X , OS X , and macOS version information Version Codename Darwin version Processor support Application support Kernel Date announced Release date Most recent version Rhapsody Developer Release Grail1Z4 / Titan1U 32 - bit PowerPC 32 - bit PowerPC 32 - bit Unknown August 31 , 1997 DR2 ( May 14 , 1998 ) Mac OS X Server 1.0 Hera Unknown March 16 , 1999 1.2 v3 ( October 27 , 2000 ) Mac OS X Developer Preview Unknown May 11 , 1998 March 16 , 1999 DP4 ( April 5 , 2000 ) Mac OS X Public Beta Kodiak 1.2. 1 Unknown September 13 , 2000 N / A Mac OS X 10.0 Cheetah 1.3. 1 Unknown March 24 , 2001 10.0. 4 ( June 22 , 2001 ) Mac OS X 10.1 Puma 1.4. 1 / 5 July 18 , 2001 September 25 , 2001 10.1. 5 ( June 6 , 2002 ) Mac OS X 10.2 Jaguar 6 32 / 64 - bit PowerPC May 6 , 2002 August 24 , 2002 10.2. 8 ( October 3 , 2003 ) Mac OS X 10.3 Panther 7 32 / 64 - bit PowerPC June 23 , 2003 October 24 , 2003 10.3. 9 ( April 15 , 2005 ) Mac OS X 10.4 Tiger 8 32 / 64 - bit PowerPC and Intel 32 / 64 - bit PowerPC and Intel May 4 , 2004 April 29 , 2005 10.4. 11 ( November 14 , 2007 ) Mac OS X 10.5 Leopard 9 32 / 64 - bit PowerPC and Intel June 26 , 2006 October 26 , 2007 10.5. 8 ( August 5 , 2009 ) Mac OS X 10.6 Snow Leopard 10 32 / 64 - bit Intel 32 / 64 - bit Intel 32 - bit PowerPC 32 / 64 - bit June 9 , 2008 August 28 , 2009 10.6. 8 v1. 1 ( July 25 , 2011 ) Mac OS X 10.7 Lion 11 64 - bit Intel 32 / 64 - bit Intel October 20 , 2010 July 20 , 2011 10.7. 5 ( September 19 , 2012 ) OS X 10.8 Mountain Lion 12 64 - bit February 16 , 2012 July 25 , 2012 10.8. 5 ( 12F45 ) ( October 3 , 2013 ) OS X 10.9 Mavericks 13 June 10 , 2013 October 22 , 2013 10.9. 5 ( 13F1112 ) ( September 18 , 2014 ) OS X 10.10 Yosemite 14 June 2 , 2014 October 16 , 2014 10.10. 5 ( 14F27 ) ( August 13 , 2015 ) OS X 10.11 El Capitan 15 June 8 , 2015 September 30 , 2015 10.11. 6 ( 15G31 ) ( July 18 , 2016 ) macOS 10.12 Sierra 16 June 13 , 2016 September 20 , 2016 10.12. 6 ( 16G29 ) ( July 19 , 2017 ) macOS 10.13 High Sierra 17 June 5 , 2017 September 25 , 2017 10.13. 4 ( 17E199 ) ( March 29 , 2018 )
  Fact 0.0: Rhapsody Developer Release had the codenames Grail1Z4 and Titan1U
  Fact 0.1: Rhapsody Developer Release supported 32-bit PowerPC processors
  Fact 0.2: Rhapsody Developer Release applications were 32-bit PowerPC
  Fact 0.3: Rhapsody Developer Release was announced August 31, 1997
  Fact 0.4: The most recent Rhapsody Developer Release version is DR2 (May 14, 1998)
  Fact 0.5: Mac OS X Server 1.0 had the codename Hera
  Fact 0.6: Mac OS X Server 1.0 was released March 16, 1999
  Fact 0.7: The most recent Mac OS X Server 1.0 version is 1.2v3 (October 27, 2000)
  Fact 0.8: Mac OS X Developer Preview was announced May 11, 1998
  Fact 0.9: The most recent Mac OS X Developer Preview version is DP4 (April 5, 2000)
  Fact 0.10: Mac OS X Public Beta had the codename Kodiak
  Fact 0.11: Mac OS X Public Beta had Darwin version 1.2.1
  Fact 0.12: Mac OS X Public Beta was released September 13, 2000
  Fact 0.13: Mac OS X Public Beta has no most recent version listed (N/A)
  Fact 0.14: Mac OS X 10.0 had the codename Cheetah
  Fact 0.15: Mac OS X 10.0 had Darwin version 1.3.1
  Fact 0.16: Mac OS X 10.0 was released March 24, 2001
  Fact 0.17: The most recent Mac OS X 10.0 version is 10.0.4 (June 22, 2001)
  Fact 0.18: Mac OS X 10.1 had the codename Puma
  Fact 0.19: Mac OS X 10.1 had Darwin version 1.4.1 / 5
  Fact 0.20: Mac OS X 10.1 was announced July 18, 2001
  Fact 0.21: Mac OS X 10.1 was released September 25, 2001
  Fact 0.22: The most recent Mac OS X 10.1 version is 10.1.5 (June 6, 2002)
  Fact 0.23: Mac OS X 10.2 had the codename Jaguar
  Fact 0.24: Mac OS X 10.2 had Darwin version 6
  Fact 0.25: Mac OS X 10.2 supported 32-bit and 64-bit PowerPC processors
  Fact 0.26: Mac OS X 10.2 was announced May 6, 2002
  Fact 0.27: Mac OS X 10.2 was released August 24, 2002
  Fact 0.28: The most recent Mac OS X 10.2 version is 10.2.8 (October 3, 2003)
  Fact 0.29: Mac OS X 10.3 had the codename Panther
  Fact 0.30: Mac OS X 10.3 had Darwin version 7
  Fact 0.31: Mac OS X 10.3 supported 32-bit and 64-bit PowerPC processors
  Fact 0.32: Mac OS X 10.3 was announced June 23, 2003
  Fact 0.33: Mac OS X 10.3 was released October 24, 2003
  Fact 0.34: The most recent Mac OS X 10.3 version is 10.3.9 (April 15, 2005)
  Fact 0.35: Mac OS X 10.4 had the codename Tiger
  Fact 0.36: Mac OS X 10.4 had Darwin version 8
  Fact 0.37: Mac OS X 10.4 supported 32-bit and 64-bit PowerPC and Intel processors
  Fact 0.38: Mac OS X 10.4 was announced May 4, 2004
  Fact 0.39: Mac OS X 10.4 was released April 29, 2005
  Fact 0.40: The most recent Mac OS X 10.4 version is 10.4.11 (November 14, 2007)
  Fact 0.41: Mac OS X 10.5 had the codename Leopard
  Fact 0.42: Mac OS X 10.5 had Darwin version 9
  Fact 0.43: Mac OS X 10.5 supported 32-bit and 64-bit PowerPC and Intel processors
  Fact 0.44: Mac OS X 10.5 was announced June 26, 2006
  Fact 0.45: Mac OS X 10.5 was released October 26, 2007
  Fact 0.46: The most recent Mac OS X 10.5 version is 10.5.8 (August 5, 2009)
  Fact 0.47: Mac OS X 10.6 had the codename Snow Leopard
  Fact 0.48: Mac OS X 10.6 had Darwin version 10
  Fact 0.49: Mac OS X 10.6 supported 32-bit and 64-bit Intel processors
  Fact 0.50: Mac OS X 10.6 supported 32-bit PowerPC processors
  Fact 0.51: Mac OS X 10.6 was announced June 9, 2008
  Fact 0.52: Mac OS X 10.6 was released August 28, 2009
  Fact 0.53: The most recent Mac OS X 10.6 version is 10.6.8 v1.1 (July 25, 2011)
  Fact 0.54: Mac OS X 10.7 had the codename Lion
  Fact 0.55: Mac OS X 10.7 had Darwin version 11
  Fact 0.56: Mac OS X 10.7 supported 64-bit Intel processors
  Fact 0.57: Mac OS X 10.7 was announced October 20, 2010
  Fact 0.58: Mac OS X 10.7 was released July 20, 2011
  Fact 0.59: The most recent Mac OS X 10.7 version is 10.7.5 (September 19, 2012)
  Fact 0.60: OS X 10.8 had the codename Mountain Lion
  Fact 0.61: OS X 10.8 had Darwin version 12
  Fact 0.62: OS X 10.8 supported 64-bit Intel processors
  Fact 0.63: OS X 10.8 was announced February 16, 2012
  Fact 0.64: OS X 10.8 was released July 25, 2012
  Fact 0.65: The most recent OS X 10.8 version is 10.8.5 (12F45) (October 3, 2013)
  Fact 0.66: OS X 10.9 had the codename Mavericks
  Fact 0.67: OS X 10.9 had Darwin version 13
  Fact 0.68: OS X 10.9 was announced June 10, 2013
  Fact 0.69: OS X 10.9 was released October 22, 2013
  Fact 0.70: The most recent OS X 10.9 version is 10.9.5 (13F1112) (September 18, 2014)
  Fact 0.71: OS X 10.10 had the codename Yosemite
  Fact 0.72: OS X 10.10 had Darwin version 14
  Fact 0.73: OS X 10.10 was announced June 2, 2014
  Fact 0.74: OS X 10.10 was released October 16, 2014
  Fact 0.75: The most recent OS X 10.10 version is 10.10.5 (14F27) (August 13, 2015)
  Fact 0.76: OS X 10.11 had the codename El Capitan
  Fact 0.77: OS X 10.11 had Darwin version 15
  Fact 0.78: OS X 10.11 was announced June 8, 2015
  Fact 0.79: OS X 10.11 was released September 30, 2015
  Fact 0.80: The most recent OS X 10.11 version is 10.11.6 (15G31) (July 18, 2016)
  Fact 0.81: macOS 10.12 had the codename Sierra
  Fact 0.82: macOS 10.12 had Darwin version 16
  Fact 0.83: macOS 10.12 was announced June 13, 2016
  Fact 0.84: macOS 10.12 was released September 20, 2016
  Fact 0.85: The most recent macOS 10.12 version is 10.12.6 (16G29) (July 19, 2017)
  Fact 0.86: macOS 10.13 had the codename High Sierra
  Fact 0.87: macOS 10.13 had Darwin version 17
  Fact 0.88: macOS 10.13 was announced June 5, 2017
  Fact 0.89: macOS 10.13 was released September 25, 2017
  Fact 0.90: The most recent macOS 10.13 version is 10.13.4 (17E199) (March 29, 2018)
Sentence 1: macOS Screenshot of the macOS High Sierra desktop Developer Apple , Inc . Written in C++ Objective - C Swift OS family Macintosh , Unix Working state Current Source model Closed source ( with open source components ) Initial release March 24 , 2001 ; 17 years ago ( 2001 - 03 - 24 ) Latest release 10.13. 4 ( 17E202 ) ( April 24 , 2018 ; 2 days ago ( 2018 - 04 - 24 ) ) ( ± ) Latest preview 10.13. 5 beta 2 ( 17B45c ) ( April 17 , 2018 ; 9 days ago ( 2018 - 04 - 17 ) ) ( ± ) Marketing target Personal computing Available in 34 languages List of languages As of macOS Sierra : Arabic , Catalan , Croatian , Chinese ( Simplified ) , Chinese ( Traditional ) , Czech , Danish , Dutch , English , Finnish , French , German , Greek , Hebrew , Hungarian , Indonesian , Italian , Japanese , Korean , Malay , Norwegian , Polish , Portuguese ( Brazil ) , Portuguese ( Portugal ) , Romanian , Russian , Slovak , Spanish ( Mexico ) , Spanish ( Spain ) , Swedish , Thai , Turkish , Ukrainian , Vietnamese Update method Mac App Store ( 10.8 + ) Software Update ( 10.0 -- 10.7 ) Platforms x86 - 64 ( 10.4. 7 - present ) IA - 32 ( 10.4. 4 -- 10.6. 8 ; discontinued ) PowerPC ( 10.0 -- 10.5. 8 ; discontinued ) Kernel type Hybrid ( XNU ) Default user interface Graphical ( Aqua ) License Commercial software , proprietary software Preceded by Classic Mac OS Official website www.apple.com/macos Support status Supported
  Fact 1.0: macOS infobox: developer is Apple Inc.
  Fact 1.1: macOS is written in C++, Objective-C, and Swift
  Fact 1.2: macOS belongs to the Macintosh and Unix OS family
  Fact 1.3: macOS is a current working system
  Fact 1.4: macOS is closed source, with open source components
  Fact 1.5: macOS initial release was March 24, 2001
  Fact 1.6: Latest macOS release is 10.13.4 (17E202), April 24, 2018
  Fact 1.7: Latest macOS preview is 10.13.5 beta 2 (17B45c), April 17, 2018
  Fact 1.8: macOS marketing target is personal computing
  Fact 1.9: As of macOS Sierra, macOS is available in 34 languages: Arabic, Catalan, Croatian, Chinese (Simplified), Chinese (Traditional), Czech, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hungarian, Indonesian, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese (Brazil), Portuguese (Portugal), Romanian, Russian, Slovak, Spanish (Mexico), Spanish (Spain), Swedish, Thai, Turkish, Ukrainian, and Vietnamese
  Fact 1.10: macOS is updated through the Mac App Store for 10.8 and later
  Fact 1.11: macOS 10.0 to 10.7 were updated through Software Update
  Fact 1.12: macOS runs on x86-64 from 10.4.7 to present
  Fact 1.13: macOS ran on IA-32 from 10.4.4 to 10.6.8, now discontinued
  Fact 1.14: macOS ran on PowerPC from 10.0 to 10.5.8, now discontinued
  Fact 1.15: macOS kernel type is hybrid (XNU)
  Fact 1.16: macOS default user interface is graphical (Aqua)
  Fact 1.17: macOS license is commercial and proprietary software
  Fact 1.18: macOS is preceded by Classic Mac OS
  Fact 1.19: macOS official website is www.apple.com/macos
  Fact 1.20: macOS support status is supported
Sentence 2: macOS is the second major series of Macintosh operating systems . The first is colloquially called the `` classic '' Mac OS , which was introduced in 1984 , and the final release of which was Mac OS 9 in 1999 . The first desktop version , Mac OS X 10.0 , was released in March 2001 , with its first update , 10.1 , arriving later that year . After this , Apple began naming its releases after big cats , which lasted until OS X 10.8 Mountain Lion . Since OS X 10.9 Mavericks , releases have been named after landmarks in California . Apple shortened the name to `` OS X '' in 2012 and then changed it to `` macOS '' in 2016 , adopting the nomenclature that it uses for their other operating systems , iOS , watchOS , and tvOS . The latest version is macOS High Sierra , which was publicly released in September 2017 .
  Fact 2.0: macOS is the second major series of Macintosh operating systems
  Fact 2.1: The first series is colloquially called the classic Mac OS
  Fact 2.2: Classic Mac OS was introduced in 1984
  Fact 2.3: The final classic Mac OS release was Mac OS 9 in 1999
  Fact 2.4: The first desktop version of macOS was Mac OS X 10.0, released in March 2001
  Fact 2.5: Mac OS X 10.1 was the first update, arriving later in 2001
  Fact 2.6: After this, Apple named its releases after big cats, until OS X 10.8 Mountain Lion
  Fact 2.7: Since OS X 10.9 Mavericks, releases have been named after landmarks in California
  Fact 2.8: Apple shortened the name to OS X in 2012
  Fact 2.9: Apple changed the name to macOS in 2016
  Fact 2.10: The name change adopted the nomenclature Apple uses for iOS, watchOS, and tvOS
  Fact 2.11: The latest version is macOS High Sierra
  Fact 2.12: macOS High Sierra was publicly released in September 2017

Retrieved items:
<item 1>
macOS, formerly Mac OS X and OS X, is Apple’s proprietary graphical operating system for Mac computers, developed since 2001. It uses the hybrid XNU kernel and
Aqua interface. Its first release, Mac OS X 10.0, launched in March 2001, followed by 10.1 later that year. It succeeded classic Mac OS, whose final version was
Mac OS 9 (1999). macOS is the world’s second-most-used desktop operating system; early versions
</item 1>

<item 2>
macOS
8 ; discontinued ) PowerPC ( 10.0 -- 10.5. 8 ; discontinued ) Kernel type Hybrid ( XNU ) Default user interface Graphical ( Aqua ) License Commercial software ,
proprietary software Preceded by Classic Mac OS Official website www.apple.com/macos Support status Supported macOS ( / ˌmækoʊˈɛs / ; previously Mac OS X , then
OS X ) is a series of graphical operating systems developed and marketed by Apple Inc. since 2001 . It is the primary operating system for Apple 's Mac family
of computers . Within the market of desktop , laptop and home computers , and by web usage , it is the second most widely used desktop OS , after Microsoft
Windows . macOS is the second major series of Macintosh operating systems . The first is colloquially called the `` classic '' Mac OS , which was introduced in
1984 , and the final release of which was Mac OS 9 in 1999 . The first desktop version , Mac OS X 10.0 , was released in March 2001 , with its first update ,
10.1 , arriving later that year .
</item 2>

<item 3>
macOS
MacOS - Wikipedia MacOS Jump to : navigation , search This article is about the current Apple operating system for Mac computers . For pre-2001 versions , see
Classic Mac OS . `` OSX '' and `` OS X '' redirect here . For other uses , see OSX ( disambiguation ) . macOS Screenshot of the macOS High Sierra desktop
Developer Apple , Inc . Written in C++ Objective - C Swift OS family Macintosh , Unix Working state Current Source model Closed source ( with open source
components ) Initial release March 24 , 2001 ; 17 years ago ( 2001 - 03 - 24 ) Latest release 10.13. 4 ( 17E202 ) ( April 24 , 2018 ; 2 days ago ( 2018 - 04 -
24 ) ) ( ± ) Latest preview 10.13.
</item 3>

<item 4>
macOS
Throughout the early 1990s , Apple had tried to create a `` next - generation '' OS to succeed its classic Mac OS through the Taligent , Copland and Gershwin
projects , but all of them were eventually abandoned . This led Apple to purchase NeXT in 1996 , allowing NeXTSTEP , then called OPENSTEP , to serve as the
basis for Apple 's next generation operating system . This purchase also led to Steve Jobs returning to Apple as an interim , and then the permanent CEO ,
shepherding the transformation of the programmer - friendly OPENSTEP into a system that would be adopted by Apple 's primary market of home users and creative
professionals . The project was first code named `` Rhapsody '' and then officially named Mac OS X . Mac OS X Launch of Mac OS X Mac OS X was originally
presented as the tenth major version of Apple 's operating system for Macintosh computers ; current versions of macOS retain the major version number `` 10 '' .
Previous Macintosh operating systems ( versions of the classic Mac OS ) were named using Arabic numerals , as with Mac OS 8 and Mac OS 9 .
</item 4>

<item 5>
macOS
. Apple . July 15 , 2004 . Retrieved December 20 , 2006 . The current version of Mac OS is Mac OS X ( pronounced `` Mac O-S ten '' ) . Jump up ^ Siracusa , John
( March 24 , 2006 ) . `` Five years of Mac OS X '' . Ars Technica . Condé Nast Digital . Retrieved April 15 , 2009 . Even Steve Jobs still says `` ecks ''
instead of `` ten '' sometimes . Jump up ^ Kelly , Spencer ( February 26 , 2011 ) . Click - BBC TV programme . BBC . Retrieved March 20 , 2011 . Of course X /
ɛks / does mean 10 , but anyone who used to poke around on Unix systems will know that in those days anything Unix had an X ( ˈɛks ) in it , and OS Ten is
written OS X ( ˈɛks ) in honour of the fact that it is based on UNIX , unlike its predecessors .
</item 5>

<item 6>
macOS, Apple’s mostly proprietary Unix-based Mac operating system, launched in 2001, succeeding classic Mac OS. Built on NeXTSTEP/OPENSTEP and the
POSIX-compliant Darwin/XNU foundation, it combines Aqua and Finder with frameworks such as Cocoa. Its evolution brought Swift, Retina support, security and
productivity features, and tighter iOS integration. HFS+ was replaced by APFS on SSDs in High Sierra (2017). Malware incidents, especially the 2011 Mac
</item 6>

<item 7>
macOS
Left to right : Cheetah / Puma ( 1 ) , Jaguar ( 2 ) , Panther ( 3 ) , Tiger ( 4 ) , Leopard ( 5 ) , Snow Leopard ( 6 ) , Lion ( 7 ) , Mountain Lion ( 8 ) ,
Mavericks ( 9 ) , Yosemite ( 10 ) , El Capitan ( 11 ) , Sierra ( 12 ) High Sierra ( 13 ) . Mac OS X , OS X , and macOS version information Version Codename
Darwin version Processor support Application support Kernel Date announced Release date Most recent version Rhapsody Developer Release Grail1Z4 / Titan1U 32 -
bit PowerPC 32 - bit PowerPC 32 - bit Unknown August 31 , 1997 DR2 ( May 14 , 1998 ) Mac OS X Server 1.0 Hera Unknown March 16 , 1999 1.2 v3 ( October 27 , 2000
) Mac OS X Developer Preview Unknown May 11 , 1998 March 16 , 1999 DP4 ( April 5 , 2000 ) Mac OS X Public Beta Kodiak 1.2.
</item 7>

<item 8>
Mac OS X, developed after a major overhaul begun in 1996, launched with the 2000 Public Beta and 2001’s slow, feature-limited Cheetah; Puma improved
performance. Jaguar and Panther added major interface and application advances, while Tiger introduced Spotlight and other features. Later releases added 64-bit
technologies, Intel support, iOS-inspired features, and annual updates. Apple’s big-cat naming shifted to California locations with Mavericks; Yosemite brought
a flatter design. “X
</item 8>

<item 9>
macOS is Apple’s current Unix-based operating system for Mac computers, succeeding the classic Mac OS era. Developed by Apple, it launched on March 24, 2001,
and is mostly closed-source, with open-source components. It is written in C++, Objective-C, and Swift. As of April 2018, the latest release was macOS High
Sierra 10.13.4.
</item 9>

<item 10>
macOS
With increasing popularity of the internet , Apple offered additional online services , including the . Mac , MobileMe and most recently iCloud products . It
later began selling third - party applications through the Mac App Store . Newer versions of Mac OS X also included modifications to the general interface ,
moving away from the striped gloss and transparency of the initial versions . Some applications began to use a brushed metal appearance , or non-pinstriped
titlebar appearance in version 10.4 . In Leopard , Apple announced a unification of the interface , with a standardized gray - gradient window style . In 2006 ,
the first Intel Macs released used a specialized version of Mac OS X 10.4 Tiger . A key development for the system was the announcement and release of the
iPhone from 2007 onwards . While Apple 's previous iPod media players used a minimal operating system , the iPhone used an operating system based on Mac OS X ,
which would later be called `` iPhone OS '' and then iOS . The simultaneous release of two operating systems based on the same frameworks placed tension on
Apple , which cited the iPhone as forcing it to delay Mac OS X 10.5 Leopard .
</item 10>
=== END ITEM id=e54c2d561047a874 ===

=== ITEM id=c9ad1922d5074b62 ===
Gold sentences and facts:
Sentence 0: Geoffrey Dyson Palmer , OBE ( born 4 June 1927 ) is an English actor known for his roles in British television sitcoms playing Jimmy Anderson in The Fall and Rise of Reginald Perrin ( 1976 -- 79 ) , Ben Parkinson in Butterflies ( 1978 -- 83 ) and Lionel Hardcastle in As Time Goes By ( 1992 -- 2005 ) . His film appearances include A Fish Called Wanda ( 1988 ) , The Madness of King George ( 1994 ) , Mrs. Brown ( 1997 ) , and Tomorrow Never Dies ( 1997 ) .
  Fact 0.0: Geoffrey Dyson Palmer, OBE, was born on 4 June 1927.
  Fact 0.1: Geoffrey Dyson Palmer holds the OBE honour.
  Fact 0.2: Geoffrey Dyson Palmer is an English actor.
  Fact 0.3: Geoffrey Dyson Palmer is known for his roles in British television sitcoms.
  Fact 0.4: Geoffrey Dyson Palmer played Jimmy Anderson in The Fall and Rise of Reginald Perrin, which aired from 1976 to 1979.
  Fact 0.5: Geoffrey Dyson Palmer played Ben Parkinson in Butterflies, which aired from 1978 to 1983.
  Fact 0.6: Geoffrey Dyson Palmer played Lionel Hardcastle in As Time Goes By, which aired from 1992 to 2005.
  Fact 0.7: Geoffrey Dyson Palmer's film appearances include A Fish Called Wanda (1988).
  Fact 0.8: Geoffrey Dyson Palmer's film appearances include The Madness of King George (1994).
  Fact 0.9: Geoffrey Dyson Palmer's film appearances include Mrs. Brown (1997).
  Fact 0.10: Geoffrey Dyson Palmer's film appearances include Tomorrow Never Dies (1997).

Retrieved items:
<item 1>
Robbie Coltrane
( 1975 ) John Hurt ( 1976 ) Derek Jacobi ( 1977 ) Peter Barkworth ( 1978 ) Edward Fox ( 1979 ) Alec Guinness ( 1980 ) Denholm Elliott ( 1981 ) Anthony Andrews (
1982 ) Alec Guinness ( 1983 ) Alan Bates ( 1984 ) Tim Pigott - Smith ( 1985 ) Bob Peck ( 1986 ) Michael Gambon ( 1987 ) David Jason ( 1988 ) Ray McAnally ( 1989
) John Thaw ( 1990 ) Ian Richardson ( 1991 ) Robert Lindsay ( 1992 ) John Thaw ( 1993 ) Robbie Coltrane ( 1994 ) Robbie Coltrane ( 1995 ) Robbie Coltrane ( 1996
) Nigel Hawthorne ( 1997 ) Simon Russell Beale ( 1998 ) Tom Courtenay ( 1999 ) Michael Gambon ( 2000 ) Michael Gambon ( 2001 ) Michael Gambon ( 2002 ) Albert
Finney ( 2003 ) Bill Nighy ( 2004 ) Rhys Ifans ( 2005 ) Mark Rylance
</item 1>

<item 2>
Geoffrey Palmer (actor)
Terence `` A Surfeit of H2O '' ( 1965 ) as Martin Smythe The Saint : `` The Rough Diamonds '' ( 1963 ) as Pete Ferguson Gideon 's Way ( TV Series ) `` The Alibi
Man '' ( 1965 ) as Jeff Grant The Baron : `` Masquerade '' ( 1966 ) as Anstruther `` The Killing '' ( 1966 ) as Anstuther The Wednesday Play : Cathy Come Home (
1966 ) as Property Agent Mrs Thursday ( 1966 ) as Henry Baxter Doctor Who : Doctor Who and the Silurians ( 1970 ) as Masters The Mutants ( 1972 ) as
Administrator `` Voyage of the Damned '' ( 2007 ) as Captain Hardaker Colditz ( 1972 ) as Doc The Fall and Rise of Reginald Perrin ( 1976 -- 1979 ) as Jimmy
Anderson Butterflies ( 1978 -- 1983 ) as Ben Parkinson The Professionals ( 1978 ) as Sinclair in `` Where The Jungle Ends '' Fawlty Towers : `` The Kipper and
the Corpse '' ( 1979 ) as Dr.
</item 2>

<item 3>
Geoffrey Palmer (actor)
as Lord Scarman The Long Walk to Finchley ( 2008 ) ; as John Crowder Agatha Christie 's Poirot : `` The Clocks '' ( 2009 ) as Vice Admiral Hamling Grandpa in My
Pocket `` Captain Dumbletwit 's Toughest Mission Yet ! '' ( 2010 ) as Grandad Gillbert Rev `` Christmas Special '' Series 2 , episode 7 ( 2011 ) as Martin Henry
IV , Part II ( 2012 ) ; as Lord Chief Justice Royal Variety Performance ( 2014 ) ; as the announcer ( voice - only ) Film ( edit ) A Prize of Arms ( 1962 ) as
Cpl . Myers Incident at Midnight ( 1963 ) as Dr. Tanfield Ring of Spies ( 1964 ) as Police Officer ( uncredited ) Cast a Giant Shadow ( 1966 ) as David (
uncredited ) O Lucky Man !
</item 3>

<item 4>
Geoffrey Palmer (actor)
Geoffrey Palmer ( actor ) - wikipedia Geoffrey Palmer ( actor ) Jump to : navigation , search Geoffrey Palmer OBE Palmer performing at A Breath of Fresh Air ,
June 2008 ( 1927 - 06 - 04 ) 4 June 1927 ( age 90 ) London , England Occupation Actor Years active 1958 -- present Spouse ( s ) Sally Green ( m . 1963 )
Children 2 , including Charles Palmer Geoffrey Dyson Palmer , OBE ( born 4 June 1927 ) is an English actor known for his roles in British television sitcoms
playing Jimmy Anderson in The Fall and Rise of Reginald Perrin ( 1976 -- 79 ) , Ben Parkinson in Butterflies ( 1978 -- 83 ) and Lionel Hardcastle in As Time
Goes By ( 1992 -- 2005 ) . His film appearances include A Fish Called Wanda ( 1988 ) , The Madness of King George ( 1994 ) , Mrs. Brown ( 1997 ) , and Tomorrow
Never Dies ( 1997 ) .
</item 4>

<item 5>
Geoffrey Palmer (actor)
Two sitcom roles brought him attention in the 1970s : the hapless brother - in - law of Reggie Perrin in The Fall and Rise of Reginald Perrin ( 1976 -- 79 ) ,
and the phlegmatic Ben Parkinson in Carla Lane 's Butterflies ( 1978 -- 83 ) . He has continued to appear in productions written by Perrin creator David Nobbs ,
the latest being the radio comedy The Maltby Collection . He starred opposite Judi Dench for over a decade in the BBC situation comedy As Time Goes By ( 1992 --
2005 ) . During this time he also appeared with Dench in other productions , including the James Bond film Tomorrow Never Dies , in which he portrayed Admiral
Roebuck , and Mrs. Brown , playing Sir Henry Ponsonby to Dench 's Queen Victoria . He played Doctor Price in the Fawlty Towers episode `` The Kipper and the
Corpse '' , determined to get breakfast amidst the confusion caused by the death of a guest and Basil 's inept way of handling the emergency .
</item 5>

<item 6>
Geoffrey Palmer (actor)
as Field Marshal Douglas Haig Inspector Morse : `` The Infernal Serpent '' ( 1990 ) as Matthew Copley - Barnes Bergerac ( 1990 ) `` Roots of Evil '' as Nigel
Carter As Time Goes By ( 1992 -- 2005 ) as Lionel Hardcastle The Legacy of Reginald Perrin ( 1996 ) as Jimmy Anderson Alice Through the Looking Glass ( 1998 )
as White King The Savages ( 2001 ) as Donald The 1940s House ( 2001 ) as Narrator Absolute Power ( 2003 ) as Lord Harcourt Grumpy Old Men ( 2003 -- 2004 , 2006
) as Narrator He Knew He Was Right ( 2004 ) as Sir Marmaduke Rowley Grumpy Old Holidays ( 2006 ) as Narrator Ashes to Ashes : `` Episode 8 '' ( 2008 ) ; as Lord
Scarman The Long Walk to Finchley ( 2008 ) ;
</item 6>

<item 7>
Robbie Coltrane
South Atlantic Raiders GLC : The Carnage Continues . The Pope Must Die Detectives on the Edge of a Nervous Breakdown Churchill : The Hollywood Years The Hunt
for Tony Blair Related articles Bad News Bad News ( album ) Bootleg 20th Century Coyote French and Saunders BAFTA TV Award for Best Actor Paul Rogers ( 1955 )
Peter Cushing ( 1956 ) Michael Gough ( 1957 ) Michael Hordern ( 1958 ) Donald Pleasence ( 1959 ) Patrick McGoohan ( 1960 ) Lee Montague ( 1961 ) Rupert Davies (
1962 ) Harry H. Corbett ( 1963 ) Alan Badel ( 1964 ) Patrick Wymark ( 1965 ) Alan Badel ( 1966 ) Warren Mitchell ( 1967 ) Eric Porter ( 1968 ) Roy Dotrice (
1969 ) Edward Woodward ( 1970 ) Keith Michell ( 1971 ) John Le Mesurier ( 1972 ) Anthony Hopkins ( 1973 ) Frank Finlay ( 1974 ) Peter Barkworth ( 1975 ) John
Hurt ( 1976 ) Derek Jacobi ( 1977 )
</item 7>

<item 8>
Stranger Things
season 5 , part II Bowen Brandt Cranston Crawford Fletcher Fraser Gunn Metzler Mitte Norris Odenkirk Paul Plemons Quezada Rankin Sane 2014 : Downton Abbey ,
series 4 Bonneville Carmichael Carter Coyle Dockery Doyle Froggatt James James - Collier Leech Logan McGovern McShera Milne Robb Smith Speleers Theobold Wilton
2015 : Downton Abbey , series 5 Bonneville Carmichael Carter Cassidy Coyle Dockery Doyle Froggatt James James - Collier Leech Logan McGovern McShera Nicol
Ovenden Smith Wilton 2016 : Stranger Things , season 1 Brown Buono Chrest Dyer Harbour Heaton Keery Matarazzo McLaughlin Modine Morgan Reynolds Ryder Schnapp
Steger Wolfhard 2017 : This Is Us , season 1 / season 2 Baker Breckenridge Brown Chavis Hartley Herman Jones Metz Moore Sullivan Ventimiglia Watson Zeile
Complete list ( 1994 -- 1999 ) ( 2000 -- 2009 ) ( 2010 -- present ) Retrieved from `` https://en.wikipedia.org/w/index.
</item 8>

<item 9>
Geoffrey Palmer (actor)
( 1973 ) as Examinator Doctor / Basil Keyes The Battle of Billy 's Pond ( 1976 ) - First Policeman The Outsider ( 1979 ) as Colonel Wyndham The Honorary Consul
( 1983 ) as Belfrage : British Ambassador A Zed & Two Noughts ( 1985 ) as Fallast Clockwise ( 1986 ) as Headmaster A Fish Called Wanda ( 1988 ) as Judge Hawks (
1988 ) as SAAB Salesman The Madness of King George ( 1994 ) as Warren Mrs. Brown ( 1997 ) as Henry Ponsonby Tomorrow Never Dies ( 1997 ) as Admiral Roebuck
Stiff Upper Lips ( 1998 ) as His Butler 's Voice Anna and the King ( 1999 ) as Lord John Bradley Rat ( 2000 ) as The Doctor Peter Pan ( 2003 ) as Sir Edward
Quiller Couch Picadilly Jim ( 2005 ) - Bayliss The Pink Panther 2 ( 2009 ) as Joubert W.E. ( 2011 ) as Stanley Baldwin Lost Christmas ( 2011 ) as Dr.
</item 9>

<item 10>
Robbie Coltrane
Sgt . Troy 1985 Defence of the Realm Leo McAskey 1986 Caravaggio Scipione 1986 Mona Lisa Thomas Eat the Rich Jeremy 1988 Fruit Machine , The The Fruit Machine
Annabelle 1989 Henry V Falstaff 1989 Bert Rigby , You 're a Fool Sid Trample 1989 Let It Ride Ticket Seller 1989 Danny , the Champion of the World Victor Hazell
1989 Slipstream Montclaire 1990 Midnight Breaks Fat 1990 Nuns on the Run Charlie McManus / Sister Inviolata 1990 Perfectly Normal Alonzo Turner 1991 Pope Must
Die , The The Pope Must Die The Pope 1991 Triple Bogey on a Par Five Hole Steffano Baccardi Oh , What a Night Todd Nominated -- Genie Award for Best Performance
by an Actor in a Supporting Role 1993 Boswell & Johnson 's Tour of the Western Isles Dr. Samuel Johnson 1993 Adventures of Huck Finn , The The Adventures of
Huck Finn Duke 1995 GoldenEye Valentin Dmitrovich Zukovsky 1997 Buddy Dr.
</item 10>
=== END ITEM id=c9ad1922d5074b62 ===

=== ITEM id=ed4e02cefd14e8dd ===
Gold sentences and facts:
Sentence 0: Roger Federer was the defending champion and successfully defended his title , defeating Marin Čilić in the final , 6 -- 2 , 6 -- 7 , 6 -- 3 , 3 -- 6 , 6 -- 1 . It was Federer 's 20th Grand Slam singles title and record - equalling sixth Australian Open men 's singles title ( tied with Roy Emerson and Novak Djokovic ) . With the win , Federer became the first male player to win at least six titles at two Grand Slam tournaments ( six at the Australian Open and eight at Wimbledon ) . Federer became the oldest man to win a Grand Slam singles title in the Open era since Ken Rosewall in 1972 . The time span between Federer 's first Grand slam glory at Wimbledon and this latest , almost 15 years , is an Open era record in the men 's singles field . This was also the 10th time that Federer has defended a Grand Slam title , with the previous time being at the 2008 US Open . Čilić became the first Croatian player to reach a singles final at the Australian Open .
  Fact 0.0: Roger Federer was the defending champion.
  Fact 0.1: Roger Federer successfully defended his title.
  Fact 0.2: Roger Federer defeated Marin Čilić in the final.
  Fact 0.3: The final score was 6-2, 6-7, 6-3, 3-6, 6-1.
  Fact 0.4: The win was Roger Federer's 20th Grand Slam singles title.
  Fact 0.5: The win was Roger Federer's record-equalling sixth Australian Open men's singles title.
  Fact 0.6: Roger Federer's sixth Australian Open men's singles title tied him with Roy Emerson and Novak Djokovic.
  Fact 0.7: With the win, Roger Federer became the first male player to win at least six titles at two Grand Slam tournaments.
  Fact 0.8: Roger Federer won six of those titles at the Australian Open.
  Fact 0.9: Roger Federer won eight of those titles at Wimbledon.
  Fact 0.10: Roger Federer became the oldest man to win a Grand Slam singles title in the Open era since Ken Rosewall won one in 1972.
  Fact 0.11: The time span between Roger Federer's first Grand Slam title at Wimbledon and his latest Grand Slam title is almost 15 years.
  Fact 0.12: That time span is an Open era record in men's singles.
  Fact 0.13: This was Roger Federer's 10th defense of a Grand Slam title.
  Fact 0.14: Roger Federer's previous Grand Slam title defense was at the 2008 US Open.
  Fact 0.15: Marin Čilić became the first Croatian player to reach a singles final at the Australian Open.
Sentence 1: Men 's Singles 2018 Australian Open Champion Roger Federer Runner - up Marin Čilić Final score 6 -- 2 , 6 -- 7 , 6 -- 3 , 3 -- 6 , 6 -- 1 Details Draw 128 ( 16 Q / 8 WC ) Seeds 32 Events Singles men women boys girls Doubles men women mixed boys girls Legends men women mixed WC Singles men women quad WC Doubles men women quad ← 2017 Australian Open 2019 →
  Fact 1.0: The men's singles champion of the 2018 Australian Open was Roger Federer.
  Fact 1.1: The men's singles runner-up of the 2018 Australian Open was Marin Čilić.
  Fact 1.2: The 2018 Australian Open men's singles final score was 6-2, 6-7, 6-3, 3-6, 6-1.
  Fact 1.3: The 2018 Australian Open men's singles draw had 128 entries (16 qualifiers and 8 wild cards).
  Fact 1.4: The 2018 Australian Open men's singles had 32 seeds.
  Fact 1.5: The previous edition was the 2017 Australian Open.
  Fact 1.6: The next edition was the 2019 Australian Open.

Retrieved items:
<item 1>
2018 Australian Open – Men's Singles
2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 2018 Australian Open men 's singles drawsheets Pre-Open era 1905 1906 1907 1908 1909 1910 1911 1912 1913
1914 1915 1916 -- 1918 ( WWI ) 1919 1920 1921 1922 1923 1924 1925 1926 1927 1928 1929 1930 1931 1932 1933 1934 1935 1936 1937 1938 1939 1940 1941 -- 1945 ( WWII
) 1946 1947 1948 1949 1950 1951 1952 1953 1954 1955 1956 1957 1958 1959 1960 1961 1962 1963 1964 1965 1966 1967 1968 Open era 1969 1970
</item 1>

<item 2>
2018 Australian Open – Men's Singles
29 January 2018 . Retrieved 7 February 2018 . Jump up ^ `` Roger Federer beats Marin Cilic to win Australian Open 2018 men 's singles final -- as it happened ''
. Guardian . 28 January 2018 . Retrieved 7 February 2018 . Jump up ^ `` From Milan To Melbourne , Chung Keeps Surging '' . atpworldtour.com. 24 January 2018 .
Retrieved 24 January 2018 . Jump up ^ Ryan , Peter ( 2018 - 01 - 23 ) . `` Hyeon Chung takes South Korean tennis to the world '' . The Sydney Morning Herald .
Retrieved 2018 - 01 - 24 . Jump up ^ `` Five takeaways from Hyeon Chung 's Aussie Open win over Novak Djokovic '' . Tennis.com. 22 January 2018 . Retrieved 23
January 2018 . Australian Open ( Open Era ) singles drawsheets Men 's singles 1969 1970 1971 1972 1973 1976 1977 Jan
</item 2>

<item 3>
2018 Australian Open – Men's Singles
1970 1971 1972 1973 1976 1977 Jan Dec 1978 1979 1980 1981 1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999 2000 2001 2002 2003 2005
2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 2018 Women 's singles 1969 1970 1971 1972 1973 1976 1977 Jan Dec 1978 1979 1980 1981 1982 1984 1985 1986
1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012
</item 3>

<item 4>
2018 Australian Open – Men's Singles
) ) Roscoe Tanner ( 1977 ( Dec ) ) Vitas Gerulaitis ( 1978 ) Guillermo Vilas ( 1979 ) Guillermo Vilas ( 1980 ) Brian Teacher ( 1981 ) Johan Kriek ( 1982 ) Johan
Kriek ( 1983 ) Mats Wilander ( 1984 ) Mats Wilander ( 1985 ) Stefan Edberg ( 1986 ) Not Held ( 1987 ) Stefan Edberg ( 1988 ) Mats Wilander ( 1989 ) Ivan Lendl (
1990 ) Ivan Lendl ( 1991 ) Boris Becker ( 1992 ) Jim Courier ( 1993 ) Jim Courier ( 1994 ) Pete Sampras ( 1995 ) Andre Agassi ( 1996 ) Boris Becker ( 1997 )
Pete Sampras ( 1998 ) Petr Korda ( 1999 ) Yevgeny Kafelnikov ( 2000 ) Andre Agassi ( 2001 ) Andre Agassi ( 2002 ) Thomas Johansson ( 2003 ) Andre Agassi ( 2004
) Roger Federer ( 2005 )
</item 4>

<item 5>
2018 Australian Open – Men's Singles
2003 ) Andre Agassi ( 2004 ) Roger Federer ( 2005 ) Marat Safin ( 2006 ) Roger Federer ( 2007 ) Roger Federer ( 2008 ) Novak Djokovic ( 2009 ) Rafael Nadal (
2010 ) Roger Federer ( 2011 ) Novak Djokovic ( 2012 ) Novak Djokovic ( 2013 ) Novak Djokovic ( 2014 ) Stanislas Wawrinka ( 2015 ) Novak Djokovic ( 2016 ) Novak
Djokovic ( 2017 ) Roger Federer ( 2018 ) Roger Federer 2018 ATP World Tour `` 2017 2019 '' Grand Slam events Australian Open ( S , D , X ) French Open ( S , D ,
X ) Wimbledon ( S , D , X ) US Open ( S , D , X ) ATP World Tour Masters 1000 Indian Wells ( S , D ) Miami ( S , D ) Monte Carlo ( S , D ) Madrid ( S , D ) Rome
( S , D ) Montreal ( S , D ) Cincinnati ( S ,
</item 5>

<item 6>
2018 Australian Open – Men's Singles
With the win , Federer became the first male player to win at least six titles at two Grand Slam tournaments ( six at the Australian Open and eight at Wimbledon
) . Federer became the oldest man to win a Grand Slam singles title in the Open era since Ken Rosewall in 1972 . The time span between Federer 's first Grand
slam glory at Wimbledon and this latest , almost 15 years , is an Open era record in the men 's singles field . This was also the 10th time that Federer has
defended a Grand Slam title , with the previous time being at the 2008 US Open . Čilić became the first Croatian player to reach a singles final at the
Australian Open . Federer and Rafael Nadal were in contention for the ATP no . 1 singles ranking at the start of the tournament . Nadal retained the top ranking
by reaching the quarterfinals .
</item 6>

<item 7>
2018 Australian Open – Men's Singles
Nadal retained the top ranking by reaching the quarterfinals . This was the first time since the 2008 Wimbledon Championships that two unseeded players ( Chung
Hyeon and Kyle Edmund ) reached the semifinals of the men 's singles event at a Grand Slam tournament , and the first time at the Australian Open since 1999 .
Chung became the first South Korean player to reach the quarterfinals and semifinals of a Grand Slam event . Djokovic 's fourth - round loss to Chung was his
first straight - sets loss at the Australian Open since 2007 . Contents ( hide ) 1 Seeds 2 Qualifying 3 Wildcards 4 Draw 4.1 Key 4.2 Finals 4.3 Top half 4.3. 1
Section 1 4.3. 2 Section 2 4.3. 3 Section 3 4.3. 4 Section 4 4.4 Bottom half 4.4. 1 Section 5 4.4. 2 Section 6 4.4. 3 Section 7 4.4.
</item 7>

<item 8>
2018 Australian Open – Men's Singles
2018 Australian Open -- men 's singles - wikipedia 2018 Australian Open -- men 's singles Jump to : navigation , search Men 's Singles 2018 Australian Open
Champion Roger Federer Runner - up Marin Čilić Final score 6 -- 2 , 6 -- 7 , 6 -- 3 , 3 -- 6 , 6 -- 1 Details Draw 128 ( 16 Q / 8 WC ) Seeds 32 Events Singles
men women boys girls Doubles men women mixed boys girls Legends men women mixed WC Singles men women quad WC Doubles men women quad ← 2017 Australian Open 2019
→ Main article : 2018 Australian Open Roger Federer was the defending champion and successfully defended his title , defeating Marin Čilić in the final , 6 -- 2
, 6 -- 7 , 6 -- 3 , 3 -- 6 , 6 -- 1 . It was Federer 's 20th Grand Slam singles title and record - equalling sixth Australian Open men 's singles title ( tied
with Roy Emerson and Novak Djokovic ) .
</item 8>

<item 9>
2018 Australian Open – Men's Singles
1965 1966 1967 1968 Open era 1969 1970 1971 1972 1973 1976 1977 1977 1978 1979 1980 1981 1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998
1999 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 2018 Australian Open men 's singles champions ( 1969 ) Rod Laver ( 1970 )
Arthur Ashe ( 1971 ) Ken Rosewall ( 1972 ) Ken Rosewall ( 1973 ) John Newcombe ( 1974 ) Jimmy Connors ( 1975 ) John Newcombe ( 1976 ) Mark Edmondson ( 1977 (
Jan ) ) Roscoe Tanner ( 1977 ( Dec ) ) Vitas Gerulaitis (
</item 9>

<item 10>
2018 Australian Open – Men's Singles
2 Section 6 4.4. 3 Section 7 4.4. 4 Section 8 5 References Seeds ( edit ) 01 . Rafael Nadal ( Quarterfinals , retired ) 02 . Roger Federer ( Champion ) 03 .
Grigor Dimitrov ( Quarterfinals ) 04 . Alexander Zverev ( Third round ) 05 . Dominic Thiem ( Fourth round ) 06 . Marin Čilić ( Final ) 07 . David Goffin (
Second round ) 08 . Jack Sock ( First round ) 09 . Stan Wawrinka ( Second round ) 10 . Pablo Carreño Busta ( Fourth round ) 11 . Kevin Anderson ( First round )
12 . Juan Martín del Potro ( Third round ) 13 . Sam Querrey ( Second round ) 14 . Novak Djokovic ( Fourth round ) 15 . Jo - Wilfried Tsonga ( Third round ) 16 .
John Isner ( First round ) 17 .
</item 10>
=== END ITEM id=ed4e02cefd14e8dd ===

=== ITEM id=a11b36fc9608495e ===
Gold sentences and facts:
Sentence 0: Sport Men Women Total Cross-country skiing 0 Total 0
  Fact 0.0: The table has columns for Sport, Men, Women and Total.
  Fact 0.1: The table lists Cross-country skiing as a sport.
  Fact 0.2: The table lists a total of 0 for Cross-country skiing.
  Fact 0.3: The table lists a grand total of 0.
Sentence 1: Bermuda competed at the 2018 Winter Olympics in Pyeongchang , South Korea , from 9 to 25 February 2018 , with one competitor . This was the nation sixth appearance in the Winter Olympics .
  Fact 1.0: Bermuda competed at the 2018 Winter Olympics.
  Fact 1.1: The 2018 Winter Olympics were held in Pyeongchang, South Korea.
  Fact 1.2: Bermuda competed from 9 to 25 February 2018.
  Fact 1.3: Bermuda had one competitor at the 2018 Winter Olympics.
  Fact 1.4: The 2018 Winter Olympics was Bermuda's sixth appearance in the Winter Olympics.

Retrieved items:
<item 1>
Bermuda sent one athlete, Tucker Murphy, to the 2018 Winter Olympics in PyeongChang, South Korea, held February 9–25. Competing in cross-country skiing, Murphy
also served as flag bearer. The Games marked Bermuda’s sixth Winter Olympics appearance; the country won no medals.
</item 1>

<item 2>
Bermuda at the 2018 Winter Olympics
Bermuda at the 2018 Winter Olympics - wikipedia Bermuda at the 2018 Winter Olympics Jump to : navigation , search Bermuda at the 2018 Winter Olympics Flag of
Bermuda IOC code BER NOC Bermuda Olympic Association Website www.olympics.bm in Pyeongchang , South Korea 9 -- 25 February 2018 Competitors 1 in 1 sport Flag
bearer Tucker Murphy Medals Gold 0 Silver 0 Bronze 0 Total 0 Winter Olympics appearances ( overview ) 1992 1994 1998 2002 2006 2014 2018 Bermuda competed at the
2018 Winter Olympics in Pyeongchang , South Korea , from 9 to 25 February 2018 , with one competitor . This was the nation sixth appearance in the Winter
Olympics . Contents ( hide ) 1 Competitors 2 Cross-country skiing 3 See also 4 References Competitors ( edit ) The following is the list of number of
competitors participating in the delegation per sport .
</item 2>

<item 3>
Bermuda sent one athlete, cross-country skier Tucker Murphy, to the 2018 Winter Olympics. Making his third consecutive Olympic appearance, Murphy competed in
the men’s 15 km freestyle, finishing 104th in 43:05.7, 9:21.8 behind the winner. Bermuda won no medals.
</item 3>

<item 4>
The source is a Wikipedia page titled “Bermuda at the 2018 Winter Olympics,” categorized under nations at the 2018 Games, Bermuda’s Winter Olympics
participation, and 2018 in Bermuda. It was last edited on 27 February 2018 and is available under the Creative Commons Attribution-ShareAlike License. The
supplied text contains no details about Bermuda’s athletes or results.
</item 4>

<item 5>
Bermuda at the 2018 Winter Olympics
Nations at the 2018 Winter Olympics in Pyeongchang , South Korea Africa Eritrea Ghana Kenya Madagascar Morocco Nigeria South Africa Togo Americas Argentina
Bermuda Bolivia Brazil Canada Chile Colombia Ecuador Jamaica Mexico Puerto Rico United States Asia China East Timor Hong Kong India Iran Japan Kazakhstan
Kyrgyzstan Lebanon Malaysia Mongolia North Korea Pakistan Philippines Singapore South Korea Chinese Taipei Thailand Uzbekistan Europe Albania Andorra Armenia
Austria Azerbaijan Belarus Belgium Bosnia and Herzegovina Bulgaria Croatia Cyprus Czech Republic Denmark Estonia Finland France Georgia Germany Great Britain
Greece Hungary Iceland Ireland Israel Italy Kosovo Latvia Liechtenstein Lithuania Luxembourg Macedonia Malta Moldova Monaco Montenegro Netherlands Norway Poland
Portugal Romania San Marino Serbia Slovakia Slovenia Spain Sweden Switzerland Turkey Ukraine Oceania Australia New Zealand Tonga Other Korea Olympic Athletes
from Russia Bermuda at the Olympics Summer Olympic Games 1936 1948 1952 1956 1960 1964 1968 1972 1976 1980 1984 1992 2000 2008 2012 2016 Winter Olympic Games
1992 1994 1998 2002 2006 2014 2018 Bermuda did not participate in 1980 due to a boycott .
</item 5>

<item 6>
Bermuda at the 2018 Winter Olympics
12 December 2017 . Retrieved 16 December 2017 . Jump up ^ `` Tucker Murphy Competes In Skiing In Iceland '' . Bernews. Hamilton , Bermuda . 14 May 2017 .
Retrieved 16 December 2017 .
</item 6>

<item 7>
Bermuda at the 2018 Winter Olympics
Retrieved from `` https://en.wikipedia.org/w/index.php?title=Bermuda_at_the_2018_Winter_Olympics&oldid=827912272 '' Categories : Nations at the 2018 Winter
Olympics Bermuda at the Winter Olympics by year 2018 in Bermuda Hidden categories : Use dmy dates from February 2018 Use British English from February 2018 Talk
Contents About Wikipedia Azərbaycanca Dansk Deutsch Eesti Español Français 한국어 हिन्दी Italiano Nederlands 日本 語 Norsk Polski Português Русский Suomi Svenska
Українська 中文 10 more Edit links This page was last edited on 27 February 2018 , at 13 : 44 . Text is available under the Creative Commons Attribution -
ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark
of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 7>

<item 8>
Bermuda was listed among nations at the 2018 Winter Olympics in Pyeongchang, South Korea. Its Winter Olympic appearances were in 1992, 1994, 1998, 2002, 2006,
2014 and 2018. Bermuda also competed in Summer Olympics from 1936 through 2016, with several gaps, and did not participate in 1980 because of the boycott.
</item 8>

<item 9>
Bermuda at the 2018 Winter Olympics
Sport Men Women Total Cross-country skiing 0 Total 0 Cross-country Skiing ( edit ) Main articles : Cross-country skiing at the 2018 Winter Olympics and
Cross-country skiing at the 2018 Winter Olympics -- Qualification Bermuda qualified one male athlete , Tucker Murphy . Murphy will be making his third straight
appearance at the Winter Olympics . Distance Athlete Event Final Time Deficit Rank Tucker Murphy Men 's 15 km freestyle 43 : 05.7 + 9 : 21.8 104 See also ( edit
) Bermuda at the 2018 Commonwealth Games References ( edit ) Jump up ^ `` Opening Ceremony Flagbearers - Olympic Winter Games , PyeongChang 2018 '' ( PDF ) .
olympic.org . International Olympic Committee ( IOC ) . 9 February 2018 . Retrieved 10 February 2018 . Jump up ^ `` Cross-country Skiing Quota List for Olympic
Games 2018 '' . www.data.fis-ski.com/ . International Ski Federation ( FIS ) . 12 December 2017 . Retrieved 16 December 2017 .
</item 9>

<item 10>
The 2018 Winter Olympics took place in PyeongChang, South Korea, February 9–25, with 102 events across 15 disciplines. New events included snowboard big air and
mixed-team competitions. North Korean athletes participated after negotiations; eligible Russians competed neutrally as Olympic Athletes from Russia following
doping sanctions, while NHL players stayed away. Alina Zagitova won women’s figure skating gold. Bermuda’s sole athlete, Tucker Murphy, finished 104th in
cross-country
</item 10>
=== END ITEM id=a11b36fc9608495e ===

=== ITEM id=5a73ef5d4863cc50 ===
Gold sentences and facts:
Sentence 0: The Earth 's magnetic field serves to deflect most of the solar wind , whose charged particles would otherwise strip away the ozone layer that protects the Earth from harmful ultraviolet radiation . One stripping mechanism is for gas to be caught in bubbles of magnetic field , which are ripped off by solar winds . Calculations of the loss of carbon dioxide from the atmosphere of Mars , resulting from scavenging of ions by the solar wind , indicate that the dissipation of the magnetic field of Mars caused a near total loss of its atmosphere .
  Fact 0.0: The Earth's magnetic field serves to deflect most of the solar wind.
  Fact 0.1: The solar wind's charged particles would otherwise strip away the ozone layer.
  Fact 0.2: The ozone layer protects the Earth from harmful ultraviolet radiation.
  Fact 0.3: One stripping mechanism is for gas to be caught in bubbles of magnetic field.
  Fact 0.4: The bubbles of magnetic field are ripped off by solar winds.
  Fact 0.5: Calculations of the loss of carbon dioxide from the atmosphere of Mars result from scavenging of ions by the solar wind.
  Fact 0.6: The calculations indicate that the dissipation of the magnetic field of Mars caused a near total loss of the atmosphere of Mars.

Retrieved items:
<item 1>
Earth's magnetic field
The geomagnetic field changes on time scales from milliseconds to millions of years . Shorter time scales mostly arise from currents in the ionosphere (
ionospheric dynamo region ) and magnetosphere , and some changes can be traced to geomagnetic storms or daily variations in currents . Changes over time scales
of a year or more mostly reflect changes in the Earth 's interior , particularly the iron - rich core . Frequently , the Earth 's magnetosphere is hit by solar
flares causing geomagnetic storms , provoking displays of aurorae . The short - term instability of the magnetic field is measured with the K - index . Data
from THEMIS show that the magnetic field , which interacts with the solar wind , is reduced when the magnetic orientation is aligned between Sun and Earth --
opposite to the previous hypothesis . During forthcoming solar storms , this could result in blackouts and disruptions in artificial satellites . Secular
variation ( edit ) Main article : Geomagnetic secular variation Estimated declination contours by year , 1590 to 1990 ( click to see variation ) . Strength of
the axial dipole component of Earth 's magnetic field from 1600 to 2020 .
</item 1>

<item 2>
Earth's magnetic field
Earth 's magnetic field Geosphere Jet stream Polar wind Earth 's magnetosphere Birkeland current Ionosphere Magnetopause Magnetosheath Magnetosphere
Magnetosphere chronology Magnetosphere particle motion Plasmasphere Ring current Van Allen radiation belt Solar wind Coronal cloud Coronal mass ejection Solar
flare Geomagnetic storm Heliosphere Interplanetary magnetic field Heliospheric current sheet Heliopause Space weather Satellites Full list Arase ( 2016 )
Cluster II Double Star Geotail IMAGE MMS ( 2015 ) Polar THEMIS Van Allen Probes WIND Research projects HAARP SHARE Unwin Radar SuperDARN Other magnetospheres
Hermian Lunar Jovian Ganymedian Saturnian Uranian Neptunian Related topics Flux tube Gas torus Lunar swirls Planetary rings Jupiter Saturn Uranus Neptune GND :
4015173 - 6 NDL : 00573127 Retrieved from `` https://en.wikipedia.org/w/index.php?title=Earth%27s_magnetic_field&oldid=819488137 '' Categories : Geomagnetism
Geophysics Magnetospheres Hidden categories : Webarchive template wayback links Good articles Commons category with local link different than on Wikidata
</item 2>

<item 3>
Earth's magnetic field
This buoyancy is enhanced by chemical separation : As the core cools , some of the molten iron solidifies and is plated to the inner core . In the process ,
lighter elements are left behind in the fluid , making it lighter . This is called compositional convection . A Coriolis effect , caused by the overall
planetary rotation , tends to organize the flow into rolls aligned along the north - south polar axis . A dynamo can amplify a magnetic field , but it needs a
`` seed '' field to get it started . For the Earth , this could have been an external magnetic field . Early in its history the Sun went through a T - Tauri
phase in which the solar wind would have had a magnetic field orders of magnitude larger than the present solar wind . However , much of the field may have been
screened out by the Earth 's mantle . An alternative source is currents in the core - mantle boundary driven by chemical reactions or variations in thermal or
electric conductivity . Such effects may still provide a small bias that are part of the boundary conditions for the geodynamo .
</item 3>

<item 4>
Earth's magnetic field
The pattern of flow is organized by the rotation of the Earth and the presence of the solid inner core . The mechanism by which the Earth generates a magnetic
field is known as a dynamo . The magnetic field is generated by a feedback loop : current loops generate magnetic fields ( Ampère 's circuital law ) ; a
changing magnetic field generates an electric field ( Faraday 's law ) ; and the electric and magnetic fields exert a force on the charges that are flowing in
currents ( the Lorentz force ) .
</item 4>

<item 5>
Earth's magnetic field
1.1 Radial dependence 6.3. 2 Global models 7 Biomagnetism 8 See also 9 References 10 Further reading 11 External links Importance ( edit ) The Earth 's magnetic
field serves to deflect most of the solar wind , whose charged particles would otherwise strip away the ozone layer that protects the Earth from harmful
ultraviolet radiation . One stripping mechanism is for gas to be caught in bubbles of magnetic field , which are ripped off by solar winds . Calculations of the
loss of carbon dioxide from the atmosphere of Mars , resulting from scavenging of ions by the solar wind , indicate that the dissipation of the magnetic field
of Mars caused a near total loss of its atmosphere . The study of past magnetic field of the Earth is known as paleomagnetism . The polarity of the Earth 's
magnetic field is recorded in igneous rocks , and reversals of the field are thus detectable as `` stripes '' centered on mid-ocean ridges where the sea floor
is spreading , while the stability of the geomagnetic poles between reversals has allowed paleomagnetists to track the past motion of continents .
</item 5>

<item 6>
Earth's magnetic field
Magnetosphere ( edit ) An artist 's rendering of the structure of a magnetosphere. 1 ) Bow shock . 2 ) Magnetosheath. 3 ) Magnetopause. 4 ) Magnetosphere. 5 )
Northern tail lobe. 6 ) Southern tail lobe. 7 ) Plasmasphere . Main article : Magnetosphere Earth 's magnetic field , predominantly dipolar at its surface , is
distorted further out by the solar wind . This is a stream of charged particles leaving the Sun 's corona and accelerating to a speed of 200 to 1000 kilometres
per second . They carry with them a magnetic field , the interplanetary magnetic field ( IMF ) . The solar wind exerts a pressure , and if it could reach Earth
's atmosphere it would erode it . However , it is kept away by the pressure of the Earth 's magnetic field . The magnetopause , the area where the pressures
balance , is the boundary of the magnetosphere .
</item 6>

<item 7>
Earth's magnetic field
Earth 's magnetic field - wikipedia Earth 's magnetic field Jump to : navigation , search Computer simulation of the Earth 's field in a period of normal
polarity between reversals . The lines represent magnetic field lines , blue when the field points towards the center and yellow when away . The rotation axis
of the Earth is centered and vertical . The dense clusters of lines are within the Earth 's core . Earth 's magnetic field , is also known as the geomagnetic
field , is the magnetic field that extends from the Earth 's interior out into space , where it meets the solar wind , a stream of charged particles emanating
from the Sun . Its magnitude at the Earth 's surface ranges from 25 to 65 microteslas ( 0.25 to 0.65 gauss ) . Roughly speaking it is the field of a magnetic
dipole currently tilted at an angle of about 11 degrees with respect to Earth 's rotational axis , as if there were a bar magnet placed at that angle at the
center of the Earth .
</item 7>

<item 8>
Earth's magnetic field
Reversals also provide the basis for magnetostratigraphy , a way of dating rocks and sediments . The field also magnetizes the crust , and magnetic anomalies
can be used to search for deposits of metal ores . Humans have used compasses for direction finding since the 11th century A.D. and for navigation since the
12th century . Although the magnetic declination does shift with time , this wandering is slow enough that a simple compass remains useful for navigation .
Using magnetoception various other organisms , ranging from some types of bacteria to pigeons , use the Earth 's magnetic field for orientation and navigation .
Main characteristics ( edit ) Description ( edit ) At any location , the Earth 's magnetic field can be represented by a three - dimensional vector . A typical
procedure for measuring its direction is to use a compass to determine the direction of magnetic North . Its angle relative to true North is the declination ( D
) or variation . Facing magnetic North , the angle the field makes with the horizontal is the inclination ( I ) or magnetic dip . The intensity ( F ) of the
field is proportional to the force it exerts on a magnet .
</item 8>

<item 9>
Earth's magnetic field
It extends several tens of thousands of kilometers into space , protecting the Earth from the charged particles of the solar wind and cosmic rays that would
otherwise strip away the upper atmosphere , including the ozone layer that protects the Earth from harmful ultraviolet radiation . Contents ( hide ) 1
Importance 2 Main characteristics 2.1 Description 2.1. 1 Intensity 2.1. 2 Inclination 2.1. 3 Declination 2.2 Geographical variation 2.3 Dipolar approximation
2.4 Magnetic poles 3 Magnetosphere 4 Time dependence 4.1 Short - term variations 4.2 Secular variation 4.3 Magnetic field reversals 4.4 Earliest appearance 4.5
Future 5 Physical origin 5.1 Earth 's core and the geodynamo 5.1. 1 Numerical models 5.2 Currents in the ionosphere and magnetosphere 6 Measurement and analysis
6.1 Detection 6.2 Crustal magnetic anomalies 6.3 Statistical models 6.3. 1 Spherical harmonics 6.3. 1.1 Radial dependence 6.3.
</item 9>

<item 10>
Earth's magnetic field
As well as deflecting the solar wind , the Earth 's magnetic field deflects cosmic rays , high - energy charged particles that are mostly from outside the Solar
system . ( Many cosmic rays are kept out of the Solar system by the Sun 's magnetosphere , or heliosphere . ) By contrast , astronauts on the Moon risk exposure
to radiation . Anyone who had been on the Moon 's surface during a particularly violent solar eruption in 2005 would have received a lethal dose . Some of the
charged particles do get into the magnetosphere . These spiral around field lines , bouncing back and forth between the poles several times per second . In
addition , positive ions slowly drift westward and negative ions drift eastward , giving rise to a ring current . This current reduces the magnetic field at the
Earth 's surface . Particles that penetrate the ionosphere and collide with the atoms there give rise to the lights of the aurorae and also emit X-rays . The
varying conditions in the magnetosphere , known as space weather , are largely driven by solar activity .
</item 10>
=== END ITEM id=5a73ef5d4863cc50 ===
