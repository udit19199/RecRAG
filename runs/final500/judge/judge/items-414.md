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

This file has 10 items.

=== ITEM id=e7c6158d227faecf ===
Gold sentences and facts:
Sentence 0: The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
  Fact 0.0: The Second Battle of Guam was fought from 21 July to 10 August 1944.
  Fact 0.1: The Second Battle of Guam was the American recapture of the Japanese-held island of Guam.
  Fact 0.2: Guam is a U.S. territory in the Mariana Islands.
  Fact 0.3: Japan captured Guam from the U.S. in the 1941 First Battle of Guam.
  Fact 0.4: The Second Battle of Guam took place during the Pacific campaign of World War II.
Sentence 1: The Battle of Antietam , also known as the Battle of Sharpsburg, particularly in the South, was fought on September 17, 1862, near Sharpsburg, Maryland and Antietam Creek as part of the Maryland Campaign.
  Fact 1.0: The Battle of Antietam was also known as the Battle of Sharpsburg, particularly in the South.
  Fact 1.1: The Battle of Antietam was fought on September 17, 1862.
  Fact 1.2: The Battle of Antietam was fought near Sharpsburg, Maryland.
  Fact 1.3: The Battle of Antietam was fought near Antietam Creek.
  Fact 1.4: The Battle of Antietam was part of the Maryland Campaign.

Retrieved items:
<item 1>
Maryland Campaign
The Maryland Campaign—or Antietam Campaign—occurred September 4–20, 1862, during the American Civil War.

 Confederate Gen. Robert E. Lee's first invasion of the North was repulsed by the Army of the Potomac under Maj. Gen. George B. McClellan, who moved to
intercept Lee and his Army of Northern Virginia and eventually attacked it near Sharpsburg, Maryland.

 The resulting Battle of Antietam was the bloodiest single-day battle in American history and is widely considered one of the major turning points of the war.
</item 1>

<item 2>
George McMillin
George Johnson McMillin (November 25, 1889 – August 29, 1983 ) was a United States Navy Rear Admiral who served as the 38th and final Naval Governor of Guam.

 He graduated from the United States Naval Academy in 1911 and served as an officer during four separate conflicts: World War I, the occupation of the Dominican
Republic, the United States occupation of Veracruz, and World War II.

 He served on the staff of both the Naval Academy and the Naval War College as well.

 He is most remembered as the commander who surrendered Guamanian forces to a much larger Japanese force during the First Battle of Guam, only the second battle
of World War II involving the United States.

 He had previously evacuated all but one civilian American citizen from the island and attempted to rebuild defenses after a strong typhoon devastated the
island the year before.

 On December 8, 1941, Japanese forces invaded Guam and McMillin surrendered two days later.

 He spent the rest of the war at various Japanese prisoner of war camps.
</item 2>

<item 3>
Battle of Antietam
The Battle of Antietam , also known as the Battle of Sharpsburg, particularly in the South, was fought on September 17, 1862, near Sharpsburg, Maryland and
Antietam Creek as part of the Maryland Campaign.

 It was the first field army–level engagement in the Eastern Theater of the American Civil War to take place on Union soil and is the bloodiest single-day
battle in American history, with a combined tally of 22,717 dead, wounded, or missing.
</item 3>

<item 4>
Battle of Attu
The Battle of Attu, which took place from 11–30 May 1943, was a battle fought between forces of the United States, aided by Canadian reconnaissance and
fighter-bomber support, and the Empire of Japan on Attu Island off the coast of the Territory of Alaska as part of the Aleutian Islands Campaign during the
American Theater and the Pacific Theater and was the only land battle of World War II fought on incorporated territory of the United States.

 It is also the only land battle in which Japanese and American forces fought in Arctic conditions.
</item 4>

<item 5>
Battle of Manila (1574)
The Battle of Manilla (1574) was a battle in the Manila area mainly in the location of what is now Parañaque between Chinese pirates, led by Limahong and the
Spanish colonial forces and their native allies.

 The battle occurred on November 29, 1574 when Limahong's fleet landed in the town of Parañaque and from there, began to assault the fortifications of
Intramuros.

 Initially, the inhabitants where disorganized and Limahong's forces routed them.

 Furthermore, the Chinese killed the Master-of-Camp of the Spanish, Martin De Goiti.

 This caused them to delay their assault on Manila as Martin de Goiti's house was an obstacle in their march.

 However, upon the arrival of a certain Filipino hero called, Galo, resistance started to organize.

 Under Galo's command, they were able to resist China-born Limahong until Mexico-born Juan de Salcedo arrived from Ilocos with 300 Ilocano Warriors and Limahong
was defeated and eventually forced to retreat.

 Thereafter Limahong abandoned his plans to invade Manila and instead, set up a temporary kingdom in Pangasinan.
</item 5>

<item 6>
Battle of Gaines's Mill
The Battle of Gaines's Mill, sometimes known as the First Battle of Cold Harbor or the Battle of Chickahominy River, took place on June 27, 1862, in Hanover
County, Virginia, as the third of the Seven Days Battles (Peninsula Campaign) of the American Civil War.

 Following the inconclusive Battle of Beaver Dam Creek (Mechanicsville) the previous day, Confederate General Robert E. Lee renewed his attacks against the
right flank of the Union Army, relatively isolated on the northern side of the Chickahominy River.

 There, Brig. Gen. Fitz John Porter's V Corps had established a strong defensive line behind Boatswain's Swamp.

 Lee's force was destined to launch the largest Confederate attack of the war, about 57,000 men in six divisions.

 Porter's reinforced V Corps held fast for the afternoon as the Confederates attacked in a disjointed manner, first with the division of Maj. Gen. A.P. Hill,
then Maj. Gen. Richard S. Ewell, suffering heavy casualties.

 The arrival of Maj. Gen. Stonewall Jackson's command was delayed, preventing the full concentration of Confederate force before Porter received some
reinforcements from the VI Corps.
</item 6>

<item 7>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 7>

<item 8>
Battle of Guam (1944)
The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands
captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
</item 8>

<item 9>
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
</item 9>

<item 10>
Battle of Manila (1898)
The Battle of Manila, sometimes called the Mock Battle of Manila, was a land engagement which took place in Manila on August 13, 1898, at the end of the
Spanish–American War, four months after the decisive victory by Commodore Dewey's Asiatic Squadron at the Battle of Manila Bay.

 The belligerents were Spanish forces led by Governor-General of the Philippines Fermín Jáudenes, and American forces led by United States Army Brigadier
General Wesley Merritt and United States Navy Commodore George Dewey.

 American forces were supported by units of the Philippine Revolutionary Army, led by Emilio Aguinaldo.
</item 10>
=== END ITEM id=e7c6158d227faecf ===

=== ITEM id=65f2ae4774b690bb ===
Gold sentences and facts:
Sentence 0: Nicolae Titulescu (] ; March 4, 1882 – March 17, 1941) was a well-known Romanian diplomat, at various times government minister, finance and foreign minister, and for two terms President of the General Assembly of the League of Nations (1930–32).
  Fact 0.0: Nicolae Titulescu was born on March 4, 1882.
  Fact 0.1: Nicolae Titulescu died on March 17, 1941.
  Fact 0.2: Nicolae Titulescu was a well-known Romanian diplomat.
  Fact 0.3: Nicolae Titulescu was at various times a government minister.
  Fact 0.4: Nicolae Titulescu was at various times finance minister.
  Fact 0.5: Nicolae Titulescu was at various times foreign minister.
  Fact 0.6: Nicolae Titulescu served for two terms as President of the General Assembly of the League of Nations.
  Fact 0.7: Nicolae Titulescu's two terms as President of the General Assembly of the League of Nations are dated 1930–32.
Sentence 1: The League of Nations (abbreviated as LN in English, "La Société des Nations " ] abbreviated as SDN or SdN in French) was an intergovernmental organisation founded on 10 January 1920 as a result of the Paris Peace Conference that ended the First World War.
  Fact 1.0: The League of Nations is abbreviated as LN in English.
  Fact 1.1: The League of Nations is abbreviated as SDN or SdN in French.
  Fact 1.2: The French name of the League of Nations is 'La Société des Nations'.
  Fact 1.3: The League of Nations was an intergovernmental organisation.
  Fact 1.4: The League of Nations was founded on 10 January 1920.
  Fact 1.5: The League of Nations was founded as a result of the Paris Peace Conference.
  Fact 1.6: The Paris Peace Conference ended the First World War.

Retrieved items:
<item 1>
Robert Suettinger
Robert Suettinger was United States President Bill Clinton's national intelligence officer for East Asia at the National Intelligence Council (NIC) from
1997-1998.

 While there, Suettinger oversaw the preparation of national intelligence estimates for the director of the Central Intelligence Agency.

 His areas of specialty included the People's Republic of China and the North Korean nuclear weapons program.

 After working in the Clinton administration, Suettinger has been a senior analyst for the Brookings Institution where he wrote the book "Beyond Tiananmen – The
Politics of U.S.-China Relations, 1989-2000".

 He also is a senior advisor at The Stimson Center.
</item 1>

<item 2>
Remus Nicolai
Remus Nicolai (b. June 10, 1977 in Bistriţa, Romania) is a retired Romanian aerobic gymnast.

 He had a successful career winning six world championships medals (one gold, two silver and three bronze) and five European championships medals (two gold, two
silver and one bronze).

 After retiring from aerobic gymnastics he opened together with his wife (Daniela Mărănducă) a private gymnastics club in Constanţa.

 As of 2013 he trains, together with his wife, the junior artistic women's National Team of Romania at the National Olympic Center in Onesti.
</item 2>

<item 3>
Line of succession to the former Romanian throne
The succession order to the throne of the Romanian monarchy, abolished since 1947, was regulated by the monarchical constitution of 1923 and the 1884 Law of the
Romanian Royal House Rules enacted pursuant to the 1866 Constitution of Romania which had confirmed the enthronement of Prince Karl (Carol) of
Hohenzollern-Sigmaringen.

 The defunct 1923 Constitution stipulated Salic law, according to which the throne was hereditary in King Carol I's legitimate descent and, if his male issue
failed, in the descent of his brothers of the Sigmaringen princely branch of the House of Hohenzollern, according to agnatic primogeniture and to the perpetual
exclusion of females and their descendants.

 The last monarch to reign in Romania was King Michael I, who was born in 1921, abdicated his throne on 30 December 1947 under coercion, and went into exile in
Switzerland.
</item 3>

<item 4>
Nicolae Titulescu
Nicolae Titulescu (] ; March 4, 1882 – March 17, 1941) was a well-known Romanian diplomat, at various times government minister, finance and foreign minister,
and for two terms President of the General Assembly of the League of Nations (1930–32).
</item 4>

<item 5>
Ioan Silviu Suciu
Ioan Silviu Suciu (born 24 November 1977 in Sibiu) is a retired Romanian artistic gymnast.

 His best event was the pommel horse.

 One of his closest rivals was his team mate and pommel horse Olympic champion Marius Urzică, who was defeated by Suciu in Ljubljana, 2004, when the latter won
the title while Urzică took 8th place.

 Suciu is an Olympic bronze medalist with the team, a silver world medalist on pommel horse and a six-time European medalist (pommel horse, vault and team).

 Suciu was one of the key team members of the Romanian gymnastics team for several years contributing to the 2004 Olympic team bronze medal and three
continental team medals (two gold and one silver).

 He also placed fourth all around at the 2004 Olympic Games.
</item 5>

<item 6>
Marius Urzică
Marius Daniel Urzică (born September 30, 1975, in Topliţa, Romania) is a Romanian gymnast.

 Urzică is an Olympic champion, a three-time world champion and a three-time European champion on pommel horse.

 He competed at three Olympic games, medaling each time on pommel horse (gold Sydney 2000, silver Atlanta 1996 and silver Athena 2004) and contributed to the
team bronze in Athens 2004.

 His unique technique and style of performance have won him the recognition as one of the greatest masters on this piece of apparatus ever, together with
Miroslav Cerar and Zoltán Magyar.

 Known as "The King of the Pommels" in 2001 he achieved the maximum score of 10.00 on this piece of apparatus at the Glasgow Grand Prix.

 Two elements in artistic gymnastics, one on pommel horse and one on parallel bars are named after him.
</item 6>

<item 7>
Theatrical Organization of Cyprus
THOC (Theatre Organization of Cyprus) (Greek: ΘΟΚ), also known as Cyprus Theatre Organisation established in 1970 is the first semi-governmental theatrical
organization in Cyprus.

 In accordance with its Founding Law (No. 71/1970) the aims and objectives of THOC are to «promote the art of theatre in Cyprus and to cultivate a sense of
theatre among the people, and to promote artistic relations between the theatre world of Cyprus and that of Greece and other countries».

 THOC is governed by a 9-member Board of Directors appointed by the President of the Republic for a 3-year term of office, and is run by a Director whose
appointment is made by the Board and ratified by the Council of Ministers.

 There is also an Artistic Committee, an advisory body responsible for artistic matters.

 Once the budget of THOC has been approved by the competent Ministry and the Council of Ministers, it is tabled for approval by the House of Representatives.

 The Organisation's policy is set out by the Board.
</item 7>

<item 8>
Graph query row:
organization: General Assembly of the League of Nations
part_of: League of Nations
founded_on: 10 January 1920
</item 8>

<item 9>
Gheorghe Tătărescu
During his first time in office, he moved closer to King Carol II, leading an ambivalent policy toward the fascist Iron Guard and ultimately becoming
instrumental in establishing the authoritarian and corporatist regime around the National Renaissance Front.

 In 1940, he accepted the cession of Bessarabia and Northern Bukovina to the Soviet Union, and consequently had to resign.
</item 9>

<item 10>
Gheorghe Tătărescu
Gheorghe I. Tătărescu (also known as "Guță Tătărescu", with a slightly antiquated pet form of his given name; 2 November 1886 – 28 March 1957) was a Romanian
politician who served 36th Prime Minister of Romania (1934–1937; 1939–1940), three times as Minister of Foreign Affairs ("interim" in 1934 and 1938; appointed
to the office in 1945-1947), and once as Minister of War (1934).

 Representing the "young liberals" faction inside the National Liberal Party (PNL), Tătărescu began his political career as a collaborator of Ion G. Duca,
becoming noted for his anti-Communism and, in time, for his conflicts with the PNL's leader Dinu Brătianu and the Foreign Minister Nicolae Titulescu.
</item 10>
=== END ITEM id=65f2ae4774b690bb ===

=== ITEM id=c9d2cccc03325948 ===
Gold sentences and facts:
Sentence 0: Darby O'Gill and the Little People is a 1959 Walt Disney Productions feature film starring Albert Sharpe, Janet Munro, Sean Connery and Jimmy O'Dea, in a tale about a wily Irishman and his battle of wits with leprechauns.
  Fact 0.0: Darby O'Gill and the Little People is a feature film.
  Fact 0.1: Darby O'Gill and the Little People was made by Walt Disney Productions.
  Fact 0.2: Darby O'Gill and the Little People was released in 1959.
  Fact 0.3: Darby O'Gill and the Little People stars Albert Sharpe.
  Fact 0.4: Darby O'Gill and the Little People stars Janet Munro.
  Fact 0.5: Darby O'Gill and the Little People stars Sean Connery.
  Fact 0.6: Darby O'Gill and the Little People stars Jimmy O'Dea.
  Fact 0.7: Darby O'Gill and the Little People tells a tale about a wily Irishman.
  Fact 0.8: The Irishman in Darby O'Gill and the Little People has a battle of wits with leprechauns.
Sentence 1:  Produced by Walt Disney Pictures, Aamir Khan Productions and UTV Motion Pictures, it stars Aamir Khan as the father.
  Fact 1.0: The film named in this sentence was produced by Walt Disney Pictures.
  Fact 1.1: The film named in this sentence was produced by Aamir Khan Productions.
  Fact 1.2: The film named in this sentence was produced by UTV Motion Pictures.
  Fact 1.3: Aamir Khan stars as the father in the film named in this sentence.

Retrieved items:
<item 1>
Darby O'Gill and the Little People
Darby O'Gill and the Little People is a 1959 Walt Disney Productions feature film starring Albert Sharpe, Janet Munro, Sean Connery and Jimmy O'Dea, in a tale
about a wily Irishman and his battle of wits with leprechauns.

 The film was directed by Robert Stevenson and its screenplay written by Lawrence Edward Watkin after the books of Herminie Templeton Kavanagh.
</item 1>

<item 2>
Graph query row:
film: Dangal
production_company: Walt Disney Pictures
</item 2>

<item 3>
List of Walt Disney Pictures films
This is a list of films released theatrically under the Walt Disney Pictures banner (known as that since 1983, with "Never Cry Wolf" as its first release) and
films released before that under the former name of the parent company, Walt Disney Productions (1929–1983).

 Most films listed here were distributed in the United States by the company's distribution division, Walt Disney Studios Motion Pictures (formerly known as
Buena Vista Distribution Company [1953–1987] and Buena Vista Pictures Distribution [1987–2007]).

 The Disney features produced before "Peter Pan" (1953) were originally distributed by RKO Radio Pictures, and are now distributed by Walt Disney Studios Motion
Pictures.
</item 3>

<item 4>
Walt Disney's Treasury of Classic Tales
Walt Disney's Treasury of Classic Tales is an American Sunday comic strip, which ran in newspapers from 13 July 1952 until 15 February 1987.

 Each story adapted a different Disney film, such as Darby O'Gill and the Little People, Peter Pan, or Davy Crockett.

 It was run in relatively few papers with 58 in 1957 and 55 in 1966 and was principally a vehicle for promoting new and re-released Disney films.
</item 4>

<item 5>
Walt Disney Pictures
Walt Disney Pictures, Inc. is an American film production company and a subsidiary of Walt Disney Studios, owned by The Walt Disney Company.

 The division is based at the Walt Disney Studios in Burbank, California, and is the main producer of live-action feature films within the Walt Disney Studios
unit.

 It took on its current name in 1983.

 Today, in conjunction with the other units of Walt Disney Studios, Walt Disney Pictures is classified as one of Hollywood's "Big Six" film studios.

 Films produced by Walt Disney Animation Studios and Pixar Animation Studios are also released under this brand.
</item 5>

<item 6>
Disney Renaissance
In the history of The Walt Disney Company, the Disney Renaissance refers to the era from 1989 to 1999 during which Walt Disney Feature Animation (renamed Walt
Disney Animation Studios in 2006) experienced a creative resurgence in producing successful animated films based on well-known stories, which restored public
and critical interest in The Walt Disney Company as a whole.

 During this era, the studio produced and released ten animated films: "The Little Mermaid" (1989), "The Rescuers Down Under" (1990), "Beauty and the Beast"
(1991), "Aladdin" (1992), "The Lion King" (1994), "Pocahontas" (1995), "The Hunchback of Notre Dame" (1996), "Hercules" (1997), "Mulan" (1998) and "Tarzan"
(1999).
</item 6>

<item 7>
Dangal (film)
Dangal (English: "Wrestling competition" ) is a 2016 Indian Hindi-language biographical sports drama film loosely based on the story of Mahavir Singh Phogat, an
amateur wrestler, who trains his daughters Geeta Phogat and Babita Kumari to be world-class wrestlers.

 It was directed by Nitesh Tiwari.

 Produced by Walt Disney Pictures, Aamir Khan Productions and UTV Motion Pictures, it stars Aamir Khan as the father.

 Both daughters go on to win medals at the 2010 Commonwealth Games.

 Babita won a silver medal at the Games, in the 51 kg class and gold at the 2014 Glasgow Games in the 55 kg class.

 In 2012, Geeta became the first Indian female wrestler to qualify for the Olympics.

 Mahavir's efforts inspired dozens of Indian women to take to wrestling.

 Fatima Sana Shaikh and Sanya Malhotra play the older selves of the sisters while Sakshi Tanwar plays their mother and Aparshakti Khurana, their cousin.
</item 7>

<item 8>
Graph query row:
film: Dangal
production_company: UTV Motion Pictures
</item 8>

<item 9>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
</item 9>

<item 10>
Graph query row:
film: Dangal
production_company: Aamir Khan Productions
</item 10>
=== END ITEM id=c9d2cccc03325948 ===

=== ITEM id=0854881a5b4fedee ===
Gold sentences and facts:
Sentence 0:  It covers an area of 1,383 sqkm south east of Melbourne, stretching from outer suburban Pakenham through rural areas to the coast at Inverloch and Phillip Island.
  Fact 0.0: The area covers 1,383 sqkm.
  Fact 0.1: The area is south east of Melbourne.
  Fact 0.2: The area stretches from outer suburban Pakenham through rural areas.
  Fact 0.3: The area stretches to the coast at Inverloch and Phillip Island.
Sentence 1:  It includes the towns of Bass, Cape Patterson, Corinella, Cowes, Grantville, Inverloch, Kilcunda, Koo Wee Rup, Lang Lang, Rhyll, San Remo, Ventnor, Wonthaggi, and part of suburban Pakenham.
  Fact 1.0: The area includes the town of Bass.
  Fact 1.1: The area includes the town of Cape Patterson.
  Fact 1.2: The area includes the town of Corinella.
  Fact 1.3: The area includes the town of Cowes.
  Fact 1.4: The area includes the town of Grantville.
  Fact 1.5: The area includes the town of Inverloch.
  Fact 1.6: The area includes the town of Kilcunda.
  Fact 1.7: The area includes the town of Koo Wee Rup.
  Fact 1.8: The area includes the town of Lang Lang.
  Fact 1.9: The area includes the town of Rhyll.
  Fact 1.10: The area includes the town of San Remo.
  Fact 1.11: The area includes the town of Ventnor.
  Fact 1.12: The area includes the town of Wonthaggi.
  Fact 1.13: The area includes part of suburban Pakenham.
Sentence 2: Pakenham is a satellite suburb of Melbourne on the edge of the West Gippsland region of Victoria, Australia, 56 km south-east of Melbourne's Central Business District, located within the Shire of Cardinia local government area.
  Fact 2.0: Pakenham is a satellite suburb of Melbourne.
  Fact 2.1: Pakenham is on the edge of the West Gippsland region of Victoria, Australia.
  Fact 2.2: Pakenham is 56 km south-east of Melbourne's Central Business District.
  Fact 2.3: Pakenham is located within the Shire of Cardinia local government area.

Retrieved items:
<item 1>
Electoral district of Broken Hill
Broken Hill was an electoral district of the Legislative Assembly in the Australian state of New South Wales.

 From 1894 to 1913, the Broken Hill district was covered by the electoral districts of "Broken Hill", Willyama and Sturt.

 In the 1912 redistribution, the electoral district of "Broken Hill" was abolished and absorbed into "Willyama" and "Sturt".

 In 1920, "Willyama", "Sturt" and the electoral district of Cobar were combined to create a three-member "Sturt".

 In 1927, single-member electorates were recreated and the city of Broken Hill was split between "Sturt" and Murray.

 From 1932, the city of Broken Hill was split between "Sturt" and Cobar.

 In 1968 "Sturt" was renamed Broken Hill and included all of the city of Broken Hill.

 In 1999, it was merged with part of "Murray" and renamed the electoral district of Murray-Darling.
</item 1>

<item 2>
Electoral district of Tullamarine
Electoral district of Tullamarine was an electoral district of the Legislative Assembly in the Australian state of Victoria.

 The district centred on the suburb of Tullamarine, about 17km north-west of Melbourne.
</item 2>

<item 3>
Beaconsfield, Victoria
Beaconsfield is a satellite suburb of Melbourne, Victoria, Australia, 46 km south-east of Melbourne's Central Business District, located within the Shire of
Cardinia local government area.

 Beaconsfield recorded a population of 6,714 at the 2016 Census.
</item 3>

<item 4>
Officer, Victoria
Officer is a satellite suburb of Melbourne, Victoria, Australia, 54 km south-east of Melbourne's Central Business District, located within the Shire of Cardinia
local government area.

 Officer recorded a population of 7,133 at the 2016 Census.
</item 4>

<item 5>
Electoral district of Emerald Hill
Emerald Hill was an electoral district of the Victorian Legislative Assembly, Australia.

 It covered part of the inner-city suburb South Melbourne and consisted part of the previous Electoral district of South Melbourne which was abolished in 1859.

 (The other part of the South Melbourne electorate became the Electoral district of Sandridge).
</item 5>

<item 6>
Electoral district of Nunawading
The electoral district of Nunawading was an electoral district of the Legislative Assembly in the Australian state of Victoria, located in the eastern Melbourne
suburb of Nunawading.

 A notable former member was future Prime Minister, Robert Menzies.
</item 6>

<item 7>
Electoral district of North Eastern Boroughs
North Eastern Boroughs was an electoral district of the Legislative Assembly in the Australian state of New South Wales from 1856 to 1859 that included the
towns of Newcastle, Stockton and Raymond Terrace.

 It was partly replaced by the electoral district of Newcastle and the electoral district of Hunter.
</item 7>

<item 8>
Pakenham, Victoria
Pakenham is a satellite suburb of Melbourne on the edge of the West Gippsland region of Victoria, Australia, 56 km south-east of Melbourne's Central Business
District, located within the Shire of Cardinia local government area.

 Pakenham recorded a population of 46,421 at the 2016 Census.
</item 8>

<item 9>
Electoral district of Braidwood
Braidwood was an electoral district of the Legislative Assembly in the Australian state of New South Wales between 1859 and 1904, which included the town of
Braidwood.

 It replaced parts of the electoral district of United Counties of Murray and St Vincent and the electoral district of Southern Boroughs.

 In 1904 it was largely absorbed into the electoral district of Queanbeyan.
</item 9>

<item 10>
Electoral district of Bass
The Electoral district of Bass is an electoral district of the Victorian Legislative Assembly.

 It covers an area of 1,383 sqkm south east of Melbourne, stretching from outer suburban Pakenham through rural areas to the coast at Inverloch and Phillip
Island.

 It includes the towns of Bass, Cape Patterson, Corinella, Cowes, Grantville, Inverloch, Kilcunda, Koo Wee Rup, Lang Lang, Rhyll, San Remo, Ventnor, Wonthaggi,
and part of suburban Pakenham.

 It lies within the Eastern Victoria Region of the upper house, the Legislative Council.
</item 10>
=== END ITEM id=0854881a5b4fedee ===

=== ITEM id=b29e98ea5e51a1ca ===
Gold sentences and facts:
Sentence 0: PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.
  Fact 0.0: PC Magazine is shortened as PCMag.
  Fact 0.1: PC Magazine is an American computer magazine.
  Fact 0.2: PC Magazine is published by Ziff Davis.
Sentence 1:  The magazine focuses on sculpture, its collectors and enthusiasts, but painting and poetry are also frequently featured.
  Fact 1.0: The magazine focuses on sculpture.
  Fact 1.1: The magazine focuses on sculpture collectors.
  Fact 1.2: The magazine focuses on sculpture enthusiasts.
  Fact 1.3: Painting is also frequently featured in the magazine.
  Fact 1.4: Poetry is also frequently featured in the magazine.

Retrieved items:
<item 1>
Windows: The Official Magazine
Windows: The Official Magazine is a technology magazine produced by Future plc in association with Microsoft, and published worldwide.

 The title describes itself as "A PC magazine for real life", and contains news, features, guides and reviews designed to show readers how to make better use of
their Windows PC.
</item 1>

<item 2>
ST/Amiga Format
ST Amiga Format was a computer magazine that covered the Atari ST and Amiga computers.

 It was published by Future plc to cover the ever growing market for the, then-new, 16-bit home computers.

 Issues were equally balanced with coverage for both Commodore Amiga and Atari ST systems.

 Issue 1 included a main feature 'ST or Amiga?

 The Choice is Yours', where the pros and cons for each machine was examined.
</item 2>

<item 3>
Sculptural Pursuit
Sculptural Pursuit was a quarterly art/literary magazine published by Hammer & Pen Productions, a Denver, Colorado publishing company.

 The magazine focuses on sculpture, its collectors and enthusiasts, but painting and poetry are also frequently featured.
</item 3>

<item 4>
Computer Magazine
"Computer Magazine" is a free publication (in addition to their webcasts and other resources) sponsored by the nonprofit UTCP (United Technology and Computing
Professionals) organization, and as such charges no fees for the publication nor is influenced by advertising, so their reviews are relied on in the industry
and considered unbiased and thorough.

 "Computer Magazine" is one of the early large technology publications and resources available on the web still existent and thriving today and that has
remained independent.

 ComputerMagazine.com is a tech news and resources consolidator that publishes part of the site in a semi-time line/blogging format that is popular among their
wide following of subscriber and non-subscriber readers, allowing readers to respond and comment on various articles.

 Site contributors include many of the well known technology authors, experts and publication sources, content and articles are provided by major technology
syndicators and by external expert technology sources (such as "Computer World", "Information Week", "Network World", "Wired," "Time", etc.)
</item 4>

<item 5>
Computer Press Association
Founded in 1983, the Computer Press Association (CPA) was established to promote excellence in the field of computer journalism.

 The association was composed of working editors, writers, producers, and freelancers who covered issues related to computers and technology.

 The CPA conducted the annual Computer Press Awards, which was the preeminent editorial awards of the computer and technology media.

 The CPA Awards honored outstanding examples in print, broadcast and electronic media.

 Awards were given for print publications, such as PC Magazine; online news media, such as Newsbytes News Network (both were multiple winners); individual
columns and features by well-known journalists such as Steven Levy (author of “”); broadcast awards such as “Best Radio Program”; as well as book awards in
categories such as Best Product Specific Book.

 CPA President Jeff Yablon (1994-1996) developed an updated code of ethics for technology journalists that was adopted by many major trade show groups, most
notably Bruno Blenheim.

 The Computer Press Association disbanded in 2000.
</item 5>

<item 6>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 6>

<item 7>
Computer Magazine
as well as "Computer Magazine" staff writers, and is currently managed and edited by the industry veteran Christopher Swearingin an MCSE and former CIO and
regarded author as well as contributor/reporter for "Computer Magazine" and other publications.
</item 7>

<item 8>
Guitar World
Guitar World is a monthly music magazine devoted to guitarists, published since July 1980.

 It contains original interviews, album and gear reviews, and guitar and bass tablature of approximately five songs each month.

 The magazine is published 13 times per year (12 monthly issues and a holiday issue).

 Formerly owned by Harris Publications, Future US bought the magazine in 2003.

 In 2012, NewBay Media bought the Music division of Future US.

 The latter company also published a spin-off title, Guitar Legends, each issue of which typically combined past articles from "Guitar World" under a specific
theme.
</item 8>

<item 9>
Shape (magazine)
Shape is a monthly English language fitness magazine started by Weider Publications in 1981, founded by Christine MacIntyre (a pioneer in women's free weight
fitness) and became the number one women's fitness magazine.

 At that time, Weider Enterprises consisted primarily of the bodybuilding magazine "Muscle & Fitness".

 Joe Weider and Christine MacIntyre had differing views of how to present "Shape", Weider endorsing a less journalistic and more commercial approach to
articles, MacIntyre endorsing a more academic, doctor-based magazine.

 Weider also endorsed a sexier approach to editorial while MacIntyre endorsed a healthier look for women, eschewing sexiness in the models and the copy.

 MacIntyre largely won that battle, editing a magazine that required that every byline have an advanced medical degree, that cover models should look healthy
rather than sexy, and that sexist language be avoided.

 Christine MacIntyre was the editor-in-chief until her death in 1988.

 Tara Kraft is the current editor-in-chief.

 "Shape" found a readership based on that formula.
</item 9>

<item 10>
Computer Magazine
Computer Magazine (and their website www.ComputerMagazine.com) is a popular magazine and online news site on computing and technology, offering current news and
reviews of popular and new business and consumer technologies, software, hardware, mobile computing, tablets, PCs, Macs, Windows, Linux, telecom, cellular,
wireless, data, cloud and science news on digital technologies and everything in the "tech-sphere and digi-verse", especially focused on information technology,
devices, software and services and related subjects, such as networking, servers, data centers and corporate data infrastructure technologies, and the Internet.

 Their online site, since 1997, is located at ComputerMagazine.com.

 "Computer Magazine" produces industry instructional and a popular ongoing webcast/podcast talk show and performs evaluations and reviews of IT industry
technology products, hardware, software and services with objective reporting widely respected as independent and objective, and trusted in the industry.
</item 10>
=== END ITEM id=b29e98ea5e51a1ca ===

=== ITEM id=7a76ad3f4ac947c9 ===
Gold sentences and facts:
Sentence 0: Andy Warhol ( ; born Andrew Warhola; August 6, 1928 – February 22, 1987) was an American artist, director and producer who was a leading figure in the visual art movement known as pop art.
  Fact 0.0: Andy Warhol was born Andrew Warhola.
  Fact 0.1: Andy Warhol's life span runs from August 6, 1928 to February 22, 1987.
  Fact 0.2: Andy Warhol was an American artist.
  Fact 0.3: Andy Warhol was an American director.
  Fact 0.4: Andy Warhol was an American producer.
  Fact 0.5: Andy Warhol was a leading figure in the visual art movement known as pop art.
Sentence 1: Alexander Butler was a British film director who made over sixty features and short films during the 1910s and 1920s including many for G.B. Samuelson's production company.
  Fact 1.0: Alexander Butler was a British film director.
  Fact 1.1: Alexander Butler made over sixty features and short films.
  Fact 1.2: Alexander Butler made those features and short films during the 1910s and 1920s.
  Fact 1.3: Alexander Butler made many of those features and short films for G.B. Samuelson's production company.

Retrieved items:
<item 1>
Herbert Lottman
Herbert Lottman (August 16, 1927, Brooklyn - August 27, 2014, Paris) was an American author who specialized in writing biographies on French subjects.

 An influential biographer, he published 17 biographies, 15 of which were related to French culture, commerce, or politics; including works on Albert Camus,
Colette, Gustave Flaubert, Henri Philippe Pétain, Jules Verne, and the Rothschild banking family of France.

 He wrote that, just before dying, Albert Camus was pledged to marry.

 Camus's estate tried to block his book, partly because of this controversial statement.
</item 1>

<item 2>
Alexander Butler
Alexander Butler was a British film director who made over sixty features and short films during the 1910s and 1920s including many for G.B. Samuelson's
production company.

 Butler directed several British films in Hollywood in 1920, where Samuelson had made an arrangement with Universal Pictures.

 Amongst his notable films are the Sherlock Holmes adaptation "The Valley of Fear" (1916) and the early British horror film "The Beetle" (1919).
</item 2>

<item 3>
Green Coca-Cola Bottles
With respect to its artistic merits, Warhol’s painting utilizes repetitive imagery in the shape of hundred and twelve almost identical bottles, evoking a sense
of mechanicalization which overlaps his former usage of mass culture objects.

 The widespread display of advertisement flooded the American public during the sixties, and Warhol had been successful in mapping it onto his canvas, no wonder
it is considered one of Warhol’s masterpieces.
</item 3>

<item 4>
Andy Warhol
Andy Warhol ( ; born Andrew Warhola; August 6, 1928 – February 22, 1987) was an American artist, director and producer who was a leading figure in the visual
art movement known as pop art.

 His works explore the relationship between artistic expression, celebrity culture, and advertising that flourished by the 1960s, and span a variety of media,
including painting, silkscreening, photography, film, and sculpture.

 Some of his best known works include the silkscreen paintings "Campbell's Soup Cans" (1962) and "Marilyn Diptych" (1962), the experimental film "Chelsea Girls"
(1966), and the multimedia events known as the "Exploding Plastic Inevitable" (1966–67).
</item 4>

<item 5>
Beautiful Darling
Beautiful Darling: The Life and Times of Candy Darling, Andy Warhol Superstar is a 2010 feature-length documentary film about Candy Darling, the transgender
pioneer, actress and Andy Warhol Superstar.

 The film was written and directed by James Rasin and features Chloë Sevigny as "the voice of Candy Darling", reading from Candy's private diaries and letters.

 Patton Oswalt voices Andy Warhol and Truman Capote.

 Louis Durra composed the score.
</item 5>

<item 6>
Green Coca-Cola Bottles
Green Coca-Cola Bottles is a 1962 painting by Andy Warhol that depicts numerous Coca-Cola bottles.

 Andy Warhol’s painting “Green Coca-Cola Bottles” attempted to take a mainstream item and converted it into a piece of art.

 Warhol’s piece was a hybrid craft that is to say that he utilized a silkscreen technique, which mechanicalized some aspects of the painting, but also kept it
so that he had to put in some of his individualized “unevenness” across the painting, to reel in the human aspect.

 The painting engenders an optimistic message for the American public, which is best described in Warhol’s own words, “What’s grand about this country is that
America started the tradition where the richest consumers buy essentially the same thing as the poorest... you can know that the President drinks Coke, Liz
Taylor drinks Coke, and, just think, you can drink Coke, too.

 A Coke is a Coke, and no amount of money can get you a better Coke.”

 Here Warhol goes to show that the democratic equality, which allowed the birth of a uniformly governed consumerist mindset, has given them an equal satiating
medium.
</item 6>

<item 7>
Exploding Plastic Inevitable
The Exploding Plastic Inevitable, sometimes simply called Plastic Inevitable or EPI, was a series of multimedia events organized by Andy Warhol between 1966 and
1967, featuring musical performances by The Velvet Underground and Nico, screenings of Warhol's films, and dancing and performances by regulars of Warhol's
Factory, especially Mary Woronov and Gerard Malanga.

 "Andy Warhol's Exploding Plastic Inevitable" is also the title of an 18-minute film by Ronald Nameth with recordings from one week of performances of the shows
which were filmed in Chicago, Illinois, in 1966.

 In December 1966 Warhol included a one-off magazine called "The Plastic Exploding Inevitable" as part of the "Aspen" No. 3 package.
</item 7>

<item 8>
John Warhola
John Warhola (May 31, 1925 – December 24, 2010) played a pivotal role in maintaining the legacy of his younger brother, pop artist Andy Warhol, assigned
responsibility by their father on his deathbed to ensure that Andy attended college and serving as a trustee of the Andy Warhol Foundation for the Visual Arts
after his brother's death in 1987.

 Warhola oversaw the establishment of The Andy Warhol Museum in Pittsburgh and the Andy Warhol Museum of Modern Art in Medzilaborce, Slovakia.
</item 8>

<item 9>
The Andy Warhol Story
The Andy Warhol Story (1966) is an underground film directed by Andy Warhol with cinematography by Paul Morrissey, and starring Edie Sedgwick and Rene Ricard
(as Andy Warhol).
</item 9>

<item 10>
I Shot Andy Warhol
I Shot Andy Warhol is a 1996 American-British independent film about the life of Valerie Solanas and her relationship with the artist Andy Warhol.

 The film marked the debut of Canadian director Mary Harron.

 The film stars Lili Taylor as Valerie, Jared Harris as Andy Warhol, and Martha Plimpton as Valerie's friend Stevie.

 Stephen Dorff plays Warhol superstar Candy Darling.

 John Cale of The Velvet Underground wrote the film's score despite protests from former band member Lou Reed.

 Yo La Tengo plays an anonymous band that is somewhat reminiscent of the group.
</item 10>
=== END ITEM id=7a76ad3f4ac947c9 ===

=== ITEM id=5a553573b79018c1 ===
Gold sentences and facts:
Sentence 0: Christopher Edward Nolan ( ; born 30 July 1970) is an English-American film director, producer, and screenwriter.
  Fact 0.0: Christopher Edward Nolan was born 30 July 1970
  Fact 0.1: Christopher Edward Nolan is English-American
  Fact 0.2: Christopher Edward Nolan is a film director
  Fact 0.3: Christopher Edward Nolan is a producer
  Fact 0.4: Christopher Edward Nolan is a screenwriter
Sentence 1: Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.
  Fact 1.0: Paul Joseph Schrader was born July 22, 1946
  Fact 1.1: Paul Joseph Schrader is an American screenwriter
  Fact 1.2: Paul Joseph Schrader is a film director
  Fact 1.3: Paul Joseph Schrader is a film critic

Retrieved items:
<item 1>
Ridley Scott
Sir Ridley Scott (born 30 November 1937) is an English film director and producer.

 Following his commercial breakthrough with the science-fiction horror film "Alien" (1979), his best known works include the neo-noir dystopian science fiction
film "Blade Runner" (1982), historical drama and Best Picture Oscar winner "Gladiator" (2000), and science fiction film "The Martian" (2015).
</item 1>

<item 2>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 2>

<item 3>
Christopher Nolan
Christopher Edward Nolan ( ; born 30 July 1970) is an English-American film director, producer, and screenwriter.

 He is one of the highest-grossing directors in history, and among the most successful and acclaimed filmmakers of the 21st century.
</item 3>

<item 4>
The Walker
The Walker is a 2007 American-British drama film written and directed by Paul Schrader.

 It is an independent production and is the latest installment in Schrader's "night workers" series of films, starting with "Taxi Driver" in 1976, followed by
"American Gigolo" in 1980 and "Light Sleeper" in 1992.
</item 4>

<item 5>
Kathryn Bigelow
Kathryn Ann Bigelow ( ; born November 27, 1951) is an American director, producer, and writer.

 Her films include the vampire Western horror film "Near Dark" (1987), the action crime film "Point Break" (1991), the science fiction action thriller "Strange
Days" (1995), the mystery thriller "The Weight of Water" (2000), the submarine thriller "" (2002), the war film "The Hurt Locker" (2008), the action thriller
war film "Zero Dark Thirty" (2012), the short film "Last Days" (2014), and the period crime drama "Detroit" (2017).

 "The Hurt Locker" won the 2009 Academy Award for Best Picture and the BAFTA Award for Best Film, and was nominated for the 2010 Golden Globe Award for Best
Drama.

 She has also acted as producer and writer for many of her films.
</item 5>

<item 6>
Danny Boyle
Danny Boyle (born 20 October 1956) is an English director, producer, screenwriter and theatre director, known for his work on films including "Shallow Grave",
"Trainspotting", "The Beach", "28 Days Later", "Sunshine", "Slumdog Millionaire", "127 Hours", and "Steve Jobs".

 His debut film "Shallow Grave" won the BAFTA Award for Best British Film.

 Boyle's 2008 film "Slumdog Millionaire" was nominated for ten Academy Awards and won eight, including the Academy Award for Best Director.

 He also won the Golden Globe and BAFTA Award for Best Director.

 Boyle was presented with the Extraordinary Contribution to Filmmaking Award at the 2008 Austin Film Festival, where he also introduced that year's AFF Audience
Award Winner "Slumdog Millionaire".
</item 6>

<item 7>
Blue Collar (film)
Blue Collar is a 1978 American crime drama film directed by Paul Schrader, in his directorial debut.

 It was written by Schrader and his brother Leonard, and stars Richard Pryor, Harvey Keitel and Yaphet Kotto.
</item 7>

<item 8>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 8>

<item 9>
John Bailey (cinematographer)
John Ira Bailey, A.S.C. (born August 10, 1942) is an American cinematographer and film director best known for his collaborations with directors Paul Schrader,
Lawrence Kasdan, Michael Apted, and Ken Kwapis.

 In August 2017, Bailey was elected President of the Academy of Motion Picture Arts and Sciences for a four-year term.
</item 9>

<item 10>
David Fincher
David Andrew Leo Fincher (born August 28, 1962) is an American director and producer, notably for films, television series and music videos.

 He was nominated for the Academy Award for Best Director for "The Curious Case of Benjamin Button" (2008) and "The Social Network" (2010).

 For the latter, he won the Golden Globe Award for Best Director and the BAFTA Award for Best Direction.
</item 10>
=== END ITEM id=5a553573b79018c1 ===

=== ITEM id=40ecfd31f07880e9 ===
Gold sentences and facts:
Sentence 0: Bumble and bumble Products, LLC is one of 27 brands owned by Estée Lauder Companies, Inc.
  Fact 0.0: Bumble and bumble Products, LLC is one of 27 brands.
  Fact 0.1: The 27 brands are owned by Estée Lauder Companies, Inc.
  Fact 0.2: Bumble and bumble Products, LLC is owned by Estée Lauder Companies, Inc.
Sentence 1:  The company owns a diverse portfolio of brands, distributed internationally through retail channels and digital commerce, and has its headquarters in Midtown Manhattan, New York City.
  Fact 1.0: Estée Lauder Companies, Inc. owns a diverse portfolio of brands.
  Fact 1.1: The brand portfolio is distributed internationally through retail channels.
  Fact 1.2: The brand portfolio is distributed internationally through digital commerce.
  Fact 1.3: Estée Lauder Companies, Inc. has its headquarters in Midtown Manhattan, New York City.

Retrieved items:
<item 1>
Bumble and bumble
Bumble and bumble Products, LLC is one of 27 brands owned by Estée Lauder Companies, Inc.

 It was founded as a New York City hair salon in 1977 by hairdresser and entrepreneur Michael Gordon.

 The company operates two flagship salons in Manhattan: a midtown location on East 56th Street and on the top two floors of The House of Bumble in the
Meatpacking District.
</item 1>

<item 2>
Dun &amp; Bradstreet
Dun & Bradstreet, Inc. () is a company that provides commercial data, analytics and insights for business, headquartered in Short Hills, a community in
Millburn, New Jersey, US.

 It offers a wide range of products, and services for risk and finance, operations and supply, and sales and marketing professionals, as well as research and
insights on global business issues, serving customers in government and industries such as communications, technology, strategic financial services, and
retail/telecommunications/manufacturing markets.

 Often referred to as D&B, the company’s database contains more than 265 million business records worldwide.
</item 2>

<item 3>
A. H. Belo
A. H. Belo Corporation is a Dallas-based media company that owns newspapers in North Texas.

 The current corporation was formed when Belo Corporation separated its broadcasting and publishing operations into two corporations.

 A. H. Belo also owns a part interest in Classified Ventures.

 Its CEO is Robert Decherd who had headed Belo before the spin-off.

 The company has its headquarters in the Belo Building in Downtown Dallas.
</item 3>

<item 4>
Universal Pictures
Universal Pictures (also referred to as Universal Studios or simply Universal) is an American film studio owned by Comcast through the Universal Filmed
Entertainment Group division of its wholly owned subsidiary NBCUniversal.

 The company was founded in 1912 by Carl Laemmle, Mark Dintenfass, Charles O. Baumann, Adam Kessel, Pat Powers, William Swanson, David Horsley, Robert H.
Cochrane, and Jules Brulatour, and is the oldest surviving film studio in the United States, the world's fourth oldest after Gaumont, Pathé and Nordisk Film,
and the oldest in terms of the overall film market .

 Its studios are located in Universal City, California, and its corporate offices are located in New York City.
</item 4>

<item 5>
Buena Vista (brand)
Buena Vista is a brand name which was historically often used for divisions and subsidiaries of The Walt Disney Company, whose primary studios, the Walt Disney
Studios, are located on Buena Vista Street in Burbank, California.

 (The studio lot is also home to the Company's corporate headquarters, the Team Disney Burbank building.)

 The logos for the various Buena Vista brands featured the "Buena Vista" wordmark superimposed over the Disney Castle image to signal the affiliation between
Buena Vista and Disney.

 The name literally means "good view" in Spanish.
</item 5>

<item 6>
Bombardier Inc.
Bombardier Inc. (] ) is a multinational aerospace and transportation company based in Montreal, Quebec, Canada.

 Starting as a maker of snow machines or snowmobiles, over the years it has grown into a large manufacturer of regional airliners, business jets, mass
transportation equipment, and recreational equipment, and a provider of financial services.
</item 6>

<item 7>
Stanley Bing
Stanley Bing is the pen name of Gil Schwartz (born May 20, 1951 in New York, NY), a business humorist and novelist.

 He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".

 He is the author of thirteen books including "What Would Machiavelli Do?"

 and "The Curriculum", a satirical textbook for a business school that also offers lessons on the Web.

 Schwartz is the senior executive vice president of corporate communications and Chief Communications Officer for CBS.
</item 7>

<item 8>
Rubin Schron
Rubin Schron, who goes by Ruby, is a New York City real estate investor, landlord and the founder of Cammeby's International Group.

 He has a personal net worth of well over $10 billion, according to data company Real Capital Analytics.

 The portfolio of Cammeby's, which Schron founded in 1967, includes office buildings, market-rate and government-subsidized apartment complexes, nursing homes,
the 16-building complex in Sunset Park now known as Industry City, a stake in the bottom half of Woolworth Building and industrial properties scattered across
Long Island.

 In 2013, Schron made an unsolicited and unsuccessful offer to buy the Empire State Building for $2 billion, but he has not had problems closing many other
deals throughout his long career.

 In 2003, an investment group led by Schron paid $600 million for a portfolio of about 6,000 outer-borough apartments from Donald Trump.
</item 8>

<item 9>
Estée Lauder Companies
The Estée Lauder Companies Inc. ( ) is an American manufacturer and marketer of prestige skincare, makeup, fragrance and hair care products.

 The company owns a diverse portfolio of brands, distributed internationally through retail channels and digital commerce, and has its headquarters in Midtown
Manhattan, New York City.
</item 9>

<item 10>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
</item 10>
=== END ITEM id=40ecfd31f07880e9 ===

=== ITEM id=e6d09690b32271f1 ===
Gold sentences and facts:
Sentence 0: Not Richard, But Dick is the seventh studio album by The Dead Milkmen, released in 1993 by Hollywood Records.
  Fact 0.0: Not Richard, But Dick is the seventh studio album by The Dead Milkmen.
  Fact 0.1: Not Richard, But Dick was released in 1993.
  Fact 0.2: Not Richard, But Dick was released by Hollywood Records.
Sentence 1: The Dead Milkmen are an American punk rock band formed in 1983 in Philadelphia, Pennsylvania.
  Fact 1.0: The Dead Milkmen are an American punk rock band.
  Fact 1.1: The Dead Milkmen were formed in 1983.
  Fact 1.2: The Dead Milkmen were formed in Philadelphia, Pennsylvania.

Retrieved items:
<item 1>
The J. Geils Band
The J. Geils Band was an American rock band formed in 1968, in Worcester, Massachusetts, under the leadership of guitarist John "J." Geils.

 The other band members included vocalist Peter Wolf, harmonica player Richard "Magic Dick" Salwitz, drummer Stephen Bladd, vocalist/keyboardist Seth Justman
and bassist Danny Klein.

 Wolf and Justman served as principal songwriters.

 The band played R&B-influenced blues rock during the 1970s and soon achieved commercial success before moving towards a more mainstream radio-friendly sound in
the early 1980s, which brought the band to its commercial peak.

 After Wolf left the band in 1983 to pursue a solo career, the band released one more album in 1984 with Justman on lead vocals before breaking up in 1985.

 Beginning in 1999, the band had several reunions prior to the death of its namesake founder J. Geils on April 11, 2017.
</item 1>

<item 2>
The Idols with Sid Vicious
The Idols with Sid Vicious is a concert album of former Sex Pistols bassist Sid Vicious performing with The Idols; though recorded in September 1978, it wasn't
released until 1993.

 Steve Dior provided a recording of this performance to the New Rose record label.

 It was originally released by the Fan Club division of New Rose.

 It was also released in Japan by Teichiku Records, under license from New Rose.
</item 2>

<item 3>
Richard Cheese
Richard Cheese & Lounge Against the Machine is a cover band and comedy act, performing popular songs in a lounge/swing style, reminiscent of Frank Sinatra, Dean
Martin, and Tony Bennett.

 Lounge singer Richard Cheese is a character created and portrayed by Los Angeles-based actor/comedian Mark Jonathan Davis.
</item 3>

<item 4>
The V.I.P.s (band)
The V.I.P.s were a British R&B musical ensemble formed in Carlisle, Cumberland, (North West England) in late 1963, out of an earlier outfit known as The
Ramrods, who had formed in Carlisle in 1960.

 After a change of personnel in April 1967, the band changed their name to "Art", and released the album "Supernatural Fairy Tales".

 They also participated to a psychedelic bizarre album called "Featuring The Human Host And The Heavy Metal Kids" by a collective known as Hapshash and the
Coloured Coat, formed by Guy Stevens and an influential British graphic design and avant-garde musical partnership between Michael English and Nigel Waymouth.

 The musicians involved in that project were Mike Harrison on keys and vocals, Luther Grosvenor on guitars, Greg Ridley on bass and Mike Kellie on drums, as
well as performances by Stevens, English and Waymouth.

 It was the first time that the term "heavy metal" was ever used in music, even though that album had nothing to do with heavy metal music, being closer to
psychedelic music.
</item 4>

<item 5>
RPA &amp; The United Nations of Sound
The band played in Australia also during the Splendour in the Grass Festival in late July.

 The band's debut album as Richard's backing lineup, entitled "United Nations of Sound", was released on 19 July 2010.

 "Born Again", the first proper single, was officially released the same day of the album as a digital download.
</item 5>

<item 6>
Not Richard, But Dick
Not Richard, But Dick is the seventh studio album by The Dead Milkmen, released in 1993 by Hollywood Records.

 After being out of print for years, Hollywood Records decided to release the album for digital download as of April 2, 2013.
</item 6>

<item 7>
Electric Guitars
Electric Guitars were an English band formed early in 1980 by Neil Davenport (vocals, lyrics) and Richard Hall (bass, vocals) who were both studying English at
Bristol University.

 The band soon increased to a five-man line-up, with Andy Saunders (guitar, vocals), Matt Salt (drums) and Dick Truscott (keyboards), they also later added two
backing singers: Sara and Wendy Patridge.

 Their first single "Health" / "Continental Shelf" was released on local label Fried Egg Records in 1980.
</item 7>

<item 8>
Seasoned Veteran
Seasoned Veteran is the third studio album by American rapper Richie Rich.

 It was released November 5, 1996 on Def Jam Recordings, and to date is the only major label release for the artist.

 The album was produced by Ali Malik, DJ Daryl, Doug Rasheed, Jermaine Dupri, Lev Berlak, Mike Mosley, Richie Rich and Rick Rock.

 It peaked at number 11 on the "Billboard" Top R&B/Hip-Hop Albums and at number 35 on the "Billboard" 200.

 Two of the singles released, "Let's Ride" and "Do G's Get To Go To Heaven?

", both appeared on multiple "Billboard" singles charts.

 The album features guest performances by 2Pac, E-40, Luniz, D'wayne Wiggins, T-Boz, and Rame Royal of Rhythm & Green.
</item 8>

<item 9>
Dick Delicious and The Tasty Testicles
Dick Delicious and The Tasty Testicles are a comedy / metal band from Atlanta, Georgia.

 The band formed in 1992 and released 3 albums between 1995 and 2003.

 In 2003 the Dick Delicious received the "Howard Stern Award for Musical Excellence" and broke up shortly afterward.
</item 9>

<item 10>
Ricky Wilson (American musician)
Ricky Helton Wilson (March 19, 1953 – October 12, 1985) was an American musician best known as the original guitarist and founding member of rock band the
B-52's.

 Born in Athens, Georgia, Wilson was the brother of fellow member Cindy Wilson.

 The B-52's was founded in 1977, when Ricky, his sister Cindy, Kate Pierson, Keith Strickland and Fred Schneider shared a tropical flaming volcano drink at a
Chinese restaurant and, after an impromptu music session at the home of their friend Owen Scott III, played for the first time at a Valentine's Day party for
friends.

.

 Wilson's unusual guitar tunings were a large contribution to the band's quirky sound .
</item 10>
=== END ITEM id=e6d09690b32271f1 ===

=== ITEM id=259535f9b9f26d65 ===
Gold sentences and facts:
Sentence 0: "Empty Orchestra" is the fourth episode of the third series of the British dark comedy anthology television programme "Inside No. 9".
  Fact 0.0: 'Empty Orchestra' is the fourth episode of the third series of Inside No. 9.
  Fact 0.1: Inside No. 9 is a British dark comedy anthology television programme.
Sentence 1:  Written by Steve Pemberton and Reece Shearsmith and directed by Guillem Morales, the episode first aired on 7 March 2017, on BBC 2.
  Fact 1.0: 'Empty Orchestra' was written by Steve Pemberton.
  Fact 1.1: 'Empty Orchestra' was written by Reece Shearsmith.
  Fact 1.2: 'Empty Orchestra' was directed by Guillem Morales.
  Fact 1.3: 'Empty Orchestra' first aired on 7 March 2017.
  Fact 1.4: 'Empty Orchestra' first aired on BBC 2.
Sentence 2:  The episode is set in a karaoke booth, and follows a group of work colleagues—Greg (Shearsmith), Fran (Sarah Hadland), Connie (Tamzin Outhwaite), Janet (Emily Howlett) and Duane (Javone Prince)—celebrating the promotion of Roger (Pemberton).
  Fact 2.0: The episode 'Empty Orchestra' is set in a karaoke booth.
  Fact 2.1: The episode 'Empty Orchestra' follows a group of work colleagues.
  Fact 2.2: The work colleague Greg is shown as Shearsmith.
  Fact 2.3: The work colleague Fran is shown as Sarah Hadland.
  Fact 2.4: The work colleague Connie is shown as Tamzin Outhwaite.
  Fact 2.5: The work colleague Janet is shown as Emily Howlett.
  Fact 2.6: The work colleague Duane is shown as Javone Prince.
  Fact 2.7: The colleagues are celebrating the promotion of Roger.
  Fact 2.8: Roger is shown as Pemberton.
Sentence 3:  Since coming to national notice playing Melanie Owen in the BBC One soap opera "EastEnders" from 1998 until 2002, she has since starred in both theatre and television productions, including army series "Red Cap" and crime drama "New Tricks".
  Fact 3.0: The performer described in this sentence came to national notice playing Melanie Owen in the BBC One soap opera EastEnders from 1998 until 2002.
  Fact 3.1: The performer has since starred in theatre productions.
  Fact 3.2: The performer has since starred in television productions.
  Fact 3.3: The performer has starred in the army series Red Cap.
  Fact 3.4: The performer has starred in the crime drama New Tricks.

Retrieved items:
<item 1>
Cold Comfort (Inside No. 9)
"Cold Comfort" is the fourth episode of the second series of the British dark comedy anthology television programme "Inside No. 9".

 The episode, which was written and directed by Steve Pemberton and Reece Shearsmith, was first broadcast on 16 April 2015 on BBC Two.

 Most of "Cold Comfort" is composed of a stream from a fixed camera on the desk of Andy, the protagonist, with smaller pictures on the side of the screen, in
the style of a CCTV feed.

 "Cold Comfort" was filmed over two and a half days in Twickenham, and was, like "A Quiet Night In" from "Inside No. 9"'s first series, highly experimental.

 It was Pemberton and Shearsmith's directorial debut.
</item 1>

<item 2>
Tamzin Outhwaite
Tamzin Maria Outhwaite ( ; born 5 November 1970) is a British actress from London.

 Since coming to national notice playing Melanie Owen in the BBC One soap opera "EastEnders" from 1998 until 2002, she has since starred in both theatre and
television productions, including army series "Red Cap" and crime drama "New Tricks".
</item 2>

<item 3>
Graph query row:
person: Tamzin Outhwaite
relationship: ACTED_IN
character: Connie
work: Empty Orchestra
broadcast_or_publication_date: 7 March 2017
</item 3>

<item 4>
Out of Control (2002 film)
Out of Control is a 2002 British television film written and directed for BBC by Dominic Savage.

 It stars Tamzin Outhwaite as Shelley Richards, an impoverished single mother whose son, Dean (Danny Young), is involved in a string of crimes that lead to his
incarceration in a young offenders institute (YOI).

 David Morrissey plays prison warden Mike, who tries to keep Dean out of trouble but has difficultly watching him all the time.

 After continuous bullying at the YOI, Dean is no longer able to cope and makes a fatal decision.

 It is the third in a loose trilogy of films by Savage about social deprivation, following "Nice Girl" (2000) and "When I Was Twelve" (2001).
</item 4>

<item 5>
Radio Cape Cod
Radio Cape Cod is a 2008 film directed and produced by Andrew Silver.

 It stars Tamzin Outhwaite as a radio interviewer coming to terms with the loss of her husband and a new love entering her life.

 She also must cope with her teenage daughter, played by Tamzin Merchant, who is experiencing her first love.

 The film also stars O. T. Fagbenle.

 The film won several festival awards.
</item 5>

<item 6>
Empty Orchestra
"Empty Orchestra" is the fourth episode of the third series of the British dark comedy anthology television programme "Inside No. 9".

 Written by Steve Pemberton and Reece Shearsmith and directed by Guillem Morales, the episode first aired on 7 March 2017, on BBC 2.

 The episode is set in a karaoke booth, and follows a group of work colleagues—Greg (Shearsmith), Fran (Sarah Hadland), Connie (Tamzin Outhwaite), Janet (Emily
Howlett) and Duane (Javone Prince)—celebrating the promotion of Roger (Pemberton).

 Rebekah Hinds also stars.
</item 6>

<item 7>
Vital Signs (TV series)
Vital Signs is a British television drama series airing on ITV from 2006.

 It stars Tamzin Outhwaite as a supermarket check-out operator who decides to become a doctor.

 The series co-stars William Beck, Fraser Ayres, Eve Best, Claudie Blakley, Lucinda Dryzek, Beth Goddard, Alfie Hunter, Brooke Kinsella, Harry Lloyd, Peter Rnic
and Steven Waddington.

 The filming of the show is based in numerous London hospitals and medical schools; predominantly the show has been shot in St George's Hospital and Medical
School.
</item 7>

<item 8>
Paradox (TV series)
Paradox is a 2009 British science fiction police drama, starring Tamzin Outhwaite as Detective Inspector Rebecca Flint.

 Written by Lizzie Mickery and produced by Clerkenwell Films for the BBC, it was filmed and set in Manchester, England.
</item 8>

<item 9>
Melanie Owen
Melanie Jane "Mel" Owen (also Healy and Beale) is a fictional character from the BBC soap opera "EastEnders", played by Tamzin Outhwaite.

 She appeared from 19 October 1998 to 12 April 2002 when Outhwaite opted to leave the role.

 Mel was introduced by executive producer Matthew Robinson and became a prominent character during her tenure in the soap.
</item 9>

<item 10>
Kathryn Drysdale
Kathryn Drysdale (born December 1981) is an English actress known for her role as Grace Shelley in The West End show "The Ruling Class" opposite James Mcavoy,
as Taylor in the movie "St Trinian's", Rhoda Swartz in Mira Nair's Vanity Fair, and Louise Brooks in the BBC comedy series "Two Pints of Lager and a Packet of
Crisps" from 2001 until her departure from the show in 2009.

 She has also starred in the fourth series of the ITV sitcom "Benidorm" and as Brownwell in William Boyd's award-winning "Any Human Heart" opposite Jim
Broadbent.

 She previously worked with Broadbent on Mira Nair's feature adaptation of William Thackery's "Vanity Fair" starring Reese Witherspoon.

 She played the role of mixed-race heiress Rhoda Swartz.

 She is also known for her role as Lizzie in the drama series "Tripping Over" by Mike Bullen.

 She appeared in "Doctor Who" as Bliss in the episode "Love & Monsters".

 She currently plays Meghan Markle in season 2 of "The Windsors".
</item 10>
=== END ITEM id=259535f9b9f26d65 ===
