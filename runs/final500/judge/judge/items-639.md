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

This file has 12 items.

=== ITEM id=664514ad76d90daa ===
Gold sentences and facts:
Sentence 0: Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and medical teaching facilities in Saint John, New Brunswick.
  Fact 0.0: Dalhousie University, commonly known as Dal, is a public research university.
  Fact 0.1: Dalhousie University is in Nova Scotia, Canada.
  Fact 0.2: Dalhousie University has three campuses in Halifax.
  Fact 0.3: Dalhousie University has a fourth campus in Bible Hill.
  Fact 0.4: Dalhousie University has medical teaching facilities in Saint John, New Brunswick.
Sentence 1: California State Polytechnic University, Pomona (CPP, Cal Poly Pomona, or Cal Poly) is a public polytechnic university located in Pomona, California, United States.
  Fact 1.0: California State Polytechnic University, Pomona is also known as CPP, Cal Poly Pomona, or Cal Poly.
  Fact 1.1: California State Polytechnic University, Pomona is a public polytechnic university.
  Fact 1.2: California State Polytechnic University, Pomona is located in Pomona, California, United States.

Retrieved items:
<item 1>
California State Polytechnic University, Pomona
California State Polytechnic University, Pomona (CPP, Cal Poly Pomona, or Cal Poly) is a public polytechnic university located in Pomona, California, United
States.

 It is one of two polytechnics in the California State University system.
</item 1>

<item 2>
Cal Poly Pomona Broncos women's basketball
The Cal Poly Pomona Broncos women's basketball team is the women's basketball team that represents California State Polytechnic University, Pomona, in Pomona,
California.

 The school's team currently competes in the California Collegiate Athletic Association.
</item 2>

<item 3>
Julian A. McPhee
Julian Aeneas McPhee (February 7, 1896 – November 10, 1967) was the sixth university president of California Polytechnic State University, San Luis Obispo (Cal
Poly SLO) from 1933 to 1966 and the first president of California State Polytechnic University, Pomona (Cal Poly Pomona) from 1938 to 1966.
</item 3>

<item 4>
Cal Poly Pomona Broncos
The Cal Poly Pomona Broncos or Cal Poly Broncos are the athletic sports teams for the California State Polytechnic University, Pomona (Cal Poly Pomona).

 Cal Poly Pomona has 10 varsity sports teams and offers student participation in a wide range of sports including soccer, volleyball, track and field,
basketball, softball, and baseball.

 Cal Poly Pomona participates at the NCAA's Division II (DII) level in the California Collegiate Athletic Association (CCAA).

 The Broncos have 65 CCAA championships and 14 NCAA National Championships.

 Current and former Cal Poly athletes have won 7 Olympic medals (3 gold, 1 silver, and 3 bronze).
</item 4>

<item 5>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 5>

<item 6>
Cal Poly Pomona University Library
The Cal Poly Pomona University Library is the main library on the campus of California State Polytechnic University, Pomona (Cal Poly Pomona).

 It has in its collection 670,580 books; 6,883 serial subscriptions and 10,417 audiovisual materials.
</item 6>

<item 7>
1981 Cal Poly Pomona Broncos football team
The 1981 Cal Poly Pomona Broncos football team represented California State Polytechnic University, Pomona during the 1981 NCAA Division II football season.

 Cal Poly Pomona competed in the California Collegiate Athletic Association (CCAA).
</item 7>

<item 8>
International Polytechnic High School
International Polytechnic High School, commonly abbreviated merely as iPoly, is a public college preparatory demonstration high school (9-12) located on the
California State Polytechnic University, Pomona (Cal Poly Pomona) campus and operated by the Los Angeles County Office of Education (LACOE) in conjunction with
the College of Education and Integrative Studies at the university.

 iPoly's curriculum is fully accredited by the Western Association of Schools and Colleges and is approved by the University of California and California State
University.

 It maintains a unique affiliation with the Los Angeles County High School for the Arts (LACHSA), which is also run by LACOE.

 Since iPoly does not lie in a fixed school district, it draws students from throughout Los Angeles, San Bernardino, Orange and Riverside counties.

 The majority of students come from the Pomona and San Gabriel valleys.

 In 2009 and 2013, iPoly was honored as a California Distinguished School by the California Department of Education.
</item 8>

<item 9>
Cal Poly Pomona College of Engineering
The Cal Poly Pomona College of Engineering is the engineering college at California State Polytechnic University, Pomona (Cal Poly Pomona or Cal Poly) located
in Pomona, California, United States.

 Well known for its "learn by doing" philosophy, the college's motto is: ""making imagination real"".

 Cal Poly has one of the top engineering college among public schools in California and, with over 5,600 students (as of fall 2015), it is also the largest
engineering college in Southern California, the second largest college of engineering in the California State University system, and the seventeenth largest
engineering college in the United States.

 In the 2017 "U.S. News & World Report" the College of Engineering is ranked the 5th best undergraduate program among public universities (11th overall) in the
West for Master's-granting universities, and "has one of the top ranked engineering programs, and graduates roughly 1 of every 14 engineers in the state of
California."
</item 9>

<item 10>
Cal Poly Pomona College of Environmental Design
The Cal Poly Pomona College of Environmental Design (CENV) is a college part of the California State Polytechnic University, Pomona (Cal Poly Pomona).

 The college houses over 1,600 students; making it one of largest environmental design programs in the United States.

 The college offers bachelor's degrees in five departments, as well as three master's degree programs.

 It is the only academic unit within the California State University system to be associated with a Pritzker Prize laureate (often referred to as "The Nobel
Prize in Architecture").
</item 10>
=== END ITEM id=664514ad76d90daa ===

=== ITEM id=a21df4db3521b165 ===
Gold sentences and facts:
Sentence 0:  Upon Victoria's death, the British throne passed to her eldest son Edward VII, a member of the House of Saxe-Coburg and Gotha through his father.
  Fact 0.0: Victoria died.
  Fact 0.1: The British throne passed to Victoria's eldest son Edward VII upon her death.
  Fact 0.2: Edward VII was a member of the House of Saxe-Coburg and Gotha through his father.
Sentence 1: The House of Saxe-Coburg and Gotha ( ; German: "") is a German dynasty that ruled the duchy of Saxe-Coburg and Gotha, which was one of the Ernestine duchies.
  Fact 1.0: The House of Saxe-Coburg and Gotha is a German dynasty.
  Fact 1.1: The House of Saxe-Coburg and Gotha ruled the duchy of Saxe-Coburg and Gotha.
  Fact 1.2: The duchy of Saxe-Coburg and Gotha was one of the Ernestine duchies.

Retrieved items:
<item 1>
Succession to the British throne
Succession to the British throne is determined by descent, gender (for people born before October 2011), legitimacy, and religion.

 Under common law, the Crown is inherited by a sovereign's children or by a childless sovereign's nearest collateral line.

 The Bill of Rights 1689 and the Act of Settlement 1701, restrict succession to the throne to the legitimate Protestant descendants of Sophia of Hanover that
are in "communion with the Church of England".

 Spouses of Roman Catholics were disqualified from 1689 until the law was amended in 2015.

 Protestant descendants of those excluded for being Roman Catholics are eligible.
</item 1>

<item 2>
Principality of Calenberg
The Principality of Calenberg was a dynastic division of the Welf duchy of Brunswick-Lüneburg established in 1432.

 Calenberg was ruled by the House of Hanover from 1635 onwards; the princes received the ninth electoral dignity of the Holy Roman Empire in 1692.

 Their territory became the nucleus of the Electorate of Hanover, ruled in personal union with the Kingdom of Great Britain from 1714 onwards.

 The principality received its name from Calenberg Castle, a residence of the Brunswick dukes.
</item 2>

<item 3>
House of Saxe-Coburg and Gotha
The House of Saxe-Coburg and Gotha ( ; German: "") is a German dynasty that ruled the duchy of Saxe-Coburg and Gotha, which was one of the Ernestine duchies.

 It is a cadet branch of the Saxon House of Wettin.
</item 3>

<item 4>
House of Hanover
The House of Hanover (or the Hanoverians ; German: "Haus Hannover" ) is a German royal dynasty that ruled the Electorate and then the Kingdom of Hanover, and
that also provided monarchs of Great Britain and Ireland from 1714 and ruled the United Kingdom until the death of Queen Victoria in 1901.

 Upon Victoria's death, the British throne passed to her eldest son Edward VII, a member of the House of Saxe-Coburg and Gotha through his father.
</item 4>

<item 5>
Duke of Bedford
Duke of Bedford (named after Bedford, England) is a title that has been created six times (for five distinct people) in the Peerage of England.

 The first and second creations came in 1414 in favour of Henry IV's third son, John, who later served as regent of France.

 He was made Earl of Kendal at the same time and was made Earl of Richmond later the same year.

 The titles became extinct on his death in 1435.

 The third creation came in 1470 in favour of George Neville, nephew of Warwick the Kingmaker.

 He was deprived of the title by Act of Parliament in 1478.

 The fourth creation came 1478 in favour of George, the third son of Edward IV.

 He died the following year at the age of two.

 The fifth creation came in 1485 in favour of Jasper Tudor, half-brother of Henry VI and uncle of Henry VII.

 He had already been created Earl of Pembroke in 1452.

 However, as he was a Lancastrian, his title was forfeited between 1461 and 1485 during the predominance of the House of York.
</item 5>

<item 6>
Ernest Augustus, King of Hanover
Ernest Augustus (5 June 1771 – 18 November 1851) was King of Hanover from 20 June 1837 until his death.

 He was the fifth son and eighth child of King George III of the United Kingdom and Hanover.

 As a fifth son, Ernest seemed unlikely to become a monarch, but none of his four elder brothers had a legitimate son who survived infancy.

 The Salic Law, which barred succession to or through a female, prevailed in Hanover; therefore, when his elder brother King William IV died in 1837, Ernest
succeeded him as King of Hanover.

 In the United Kingdom the succession to the monarchy was determined by primogeniture and his niece Victoria succeeded to the throne, thus ending the personal
union between the British Isles and Hanover that had existed since 1714.
</item 6>

<item 7>
House of Oldenburg
The House of Oldenburg is a European dynasty of North German origin.

 It is one of Europe's most influential royal houses, with branches that rule or have ruled in Denmark, Iceland, Greece, Norway, Russia, Sweden, Schleswig,
Holstein, and Oldenburg.

 The current Queen of Denmark and King of Norway, the former King of Greece, the consorts of Greece and the United Kingdom, as well as the first twelve names in
the line of succession to the British throne, are all patrilineal members of the Glücksburg branch of this house.
</item 7>

<item 8>
Canada under British rule
Although confederation in 1867 led to an enlarged Dominion with increased autonomy over domestic affairs, Canada still remained a colony within the British
Empire and was thus subordinate to the British Parliament until the enactment of the Statute of Westminster in 1931.

 This statute recognized Canada as an independent peer coequal with the United Kingdom, and thus provided the Parliament of Canada with legislative sovereignty
over all federal matters except the power to change the constitutional laws of Canada which remained under the purview of the Parliament of the United Kingdom.

 Canada's final vestige of legal dependence on the United Kingdom was terminated in 1982 with the enactment of the Canada Act, subsequently providing Canada
with full legal sovereignty completely independent of the United Kingdom.
</item 8>

<item 9>
Prince Christian of Hanover (born 1985)
Prince Christian of Hanover (Christian Heinrich Clemens Paul Frank Peter Welf Wilhelm-Ernst Friedrich Franz Prince of Hanover, Duke of Brunswick and Lüneburg;
born 1 June 1985) is the younger son of Ernst August, Prince of Hanover, and his first wife Chantal Hochuli.

 He is the second in the line of succession to the former Hanoverian throne, after his elder brother Prince Ernst August.

 As a descendant of George III of the United Kingdom, Christian is also in the line of succession to the British throne.
</item 9>

<item 10>
The entries survey monarchs, dynasties, and succession systems. Hawaiʻi’s Kalākaua, its last king, revived public hula and Hawaiian culture. Hanover ruled
Britain from 1714 until Victoria’s death; Hanover and Oldenburg branches remain linked to European royal families. Calenberg became Hanover’s core, while
Hanover’s Ernest Augustus inherited its throne in 1837 under Salic law. Other topics include Silesian Bernstadt, Tus
</item 10>
=== END ITEM id=a21df4db3521b165 ===

=== ITEM id=30958c1371a4c00b ===
Gold sentences and facts:
Sentence 0: The 1997 European Athletics Junior Championships were held in Ljubljana, Slovenia on July 24–27.
  Fact 0.0: The 1997 European Athletics Junior Championships were held in Ljubljana, Slovenia.
  Fact 0.1: The 1997 European Athletics Junior Championships were held on July 24-27.
Sentence 1: Ljubljana (] , ] ; also known by other, historical names) is the capital and largest city of Slovenia.
  Fact 1.0: Ljubljana is the capital of Slovenia.
  Fact 1.1: Ljubljana is the largest city of Slovenia.
  Fact 1.2: Ljubljana is also known by other, historical names.

Retrieved items:
<item 1>
European 10,000m Cup
The European 10,000m Cup is an annual 10,000 metres race for European athletes which was first held in 1997.

 The competition is organised by the European Athletics Association and first began as the European 10000 Metres Challenge after the event was removed from the
European Cup programme.

 The competition has roots in the Iberian 10,000 metres Championships – a competition between Spanish and Portuguese athletes that was held between 1991 and
1996 – and the first five editions of the European 10000 Metres Challenge were held in the Iberian Peninsula.

 The event was first held under its current title in 2005.
</item 1>

<item 2>
Ljubljana
Ljubljana (] , ] ; also known by other, historical names) is the capital and largest city of Slovenia.

 It has been the cultural, educational, economic, political, and administrative center of independent Slovenia since 1991.

 Its central geographic location within Slovenia, transport connections, concentration of industry, scientific and research institutions, and cultural tradition
are contributing factors to its leading position.
</item 2>

<item 3>
1964 European Junior Games
The 1964 European Junior Games was the first edition of the biennial athletics competition for European athletes aged under twenty.

 It was an unofficial competition without sanction from the European Athletic Association.

 The event was held at the 10th-Anniversary Stadium in Warsaw, Poland.

 The competition eventually led to the creation of the European Athletics Junior Championships in 1970.
</item 3>

<item 4>
1996 IPB Czech Indoor – Doubles
Jonas Björkman and Javier Frana was the defending champions but did not compete that year.
</item 4>

<item 5>
The European 10,000m Cup, organized annually by European Athletics since 1997, grew from the Iberian championships and was renamed in 2005. Track 10,000m races
cover 25 laps; the 1500m is a major Olympic middle-distance event. Charlottesville’s annual Bill Steers Men’s 4-Miler, held since 2004, promotes men’s health.
European under-20 championships evolved from junior games (1964, 1968)
</item 5>

<item 6>
Amela Terzić
Amela Terzić (Serbian Cyrillic: Амела Терзић, born 2 January 1993 in Priboj, FR Yugoslavia) is a Serbian middle-distance runner.

 She won two gold medals at 2011 European Athletics Junior Championships in Tallinn and was the junior champion at the 2012 European Cross Country
Championships.

 She has also been a medallist in the 1500 metres at the World Junior and World Youth Championships and a gold medalist at the 2013 European U23 Championship.
</item 6>

<item 7>
2015 European Athletics Junior Championships
The 2015 European Athletics Junior Championships were the 23rd edition of the biennial European U20 athletics championships.

 They were held in Eskilstuna, Sweden from 16 July to 19 July.
</item 7>

<item 8>
1997 European Athletics Junior Championships
The 1997 European Athletics Junior Championships were held in Ljubljana, Slovenia on July 24–27.
</item 8>

<item 9>
European Athletics U20 Championships
The European Athletics U20 Championships (formerly named the European Athletics Junior Championships up to 2015) are the European championships for athletes who
are 19 years of age or under, which is the age range recognised by the IAAF as junior athletes.

 The event is currently organized by the European Athletic Association.
</item 9>

<item 10>
1968 European Junior Games
The 1968 European Junior Games was the third edition of the biennial athletics competition for European athletes aged under twenty.

 It was the final edition of the competition under that name and was succeeded by the European Athletics Junior Championships in 1970.

 The event was held in Leipzig, part of the German Democratic Republic at that time.

 The competition was mainly between Eastern European countries, with Belgium, Austria, Netherlands and Greece being present from the west.
</item 10>
=== END ITEM id=30958c1371a4c00b ===

=== ITEM id=5ee90058fdf8d1f4 ===
Gold sentences and facts:
Sentence 0: The Battle of Okinawa (Japanese: 沖縄戦 , Hepburn: Okinawa-sen ) (Okinawan: 沖縄戦 , "Uchinaa ikusa " ), codenamed Operation Iceberg, was a major battle of the Pacific War fought on the island of Okinawa by United States Marine and Army forces against the Imperial Japanese Army and included the largest amphibious assault in the Pacific War during World War II, the April 1, 1945, invasion of Okinawa itself.
  Fact 0.0: The Battle of Okinawa is also known as 沖縄戦 (Okinawa-sen in Hepburn romanization) and as 'Uchinaa ikusa' in Okinawan.
  Fact 0.1: The Battle of Okinawa was codenamed Operation Iceberg.
  Fact 0.2: The Battle of Okinawa was a major battle of the Pacific War.
  Fact 0.3: The Battle of Okinawa was fought on the island of Okinawa.
  Fact 0.4: The Battle of Okinawa was fought by United States Marine and Army forces against the Imperial Japanese Army.
  Fact 0.5: The Battle of Okinawa included the largest amphibious assault in the Pacific War during World War II.
  Fact 0.6: The largest amphibious assault in the Pacific War was the April 1, 1945 invasion of Okinawa itself.
Sentence 1: Operation Diadem, also referred to as the Fourth Battle of Monte Cassino or, in Canada, the Battle of the Liri Valley, was an offensive operation undertaken by the Allies of World War II (U.S. Fifth Army and British Eighth Army in May 1944, as part of the Italian Campaign of World War II.
  Fact 1.0: Operation Diadem was also referred to as the Fourth Battle of Monte Cassino.
  Fact 1.1: In Canada, Operation Diadem was called the Battle of the Liri Valley.
  Fact 1.2: Operation Diadem was an offensive operation undertaken by the Allies of World War II.
  Fact 1.3: The Allies in Operation Diadem included the U.S. Fifth Army.
  Fact 1.4: The Allies in Operation Diadem included the British Eighth Army.
  Fact 1.5: Operation Diadem took place in May 1944.
  Fact 1.6: Operation Diadem was part of the Italian Campaign of World War II.

Retrieved items:
<item 1>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 1>

<item 2>
Battle of Guam (1944)
The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands
captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
</item 2>

<item 3>
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
</item 3>

<item 4>
Battle of Bataan
The Battle of Bataan (7 January – 9 April 1942) represented the most intense phase of Imperial Japan's invasion of the Philippines during World War II.

 In January 1942, forces of the Imperial Japanese Army and Navy invaded Luzon along with several islands in the Philippine Archipelago after the bombing of the
American naval base at Pearl Harbor.

 The commander-in-chief of all Filipino and American forces in the islands, General Douglas MacArthur, consolidated all of his Luzon-based units on the Bataan
Peninsula to fight against the Japanese invaders.

 By this time, the Japanese controlled nearly all of Southeast Asia.

 The Bataan peninsula and the island of Corregidor were the only remaining Allied strongholds in the region.

 Despite a lack of supplies, Filipino and American forces managed to fight the Japanese for three months, engaging them initially in a fighting retreat
southward.

 As the combined Filipino and American forces made a last stand, the delay cost the Japanese valuable time and prevented immediate victory across the Pacific.
</item 4>

<item 5>
Operation Kikusui
Operation Kikusui (菊水作戦 , Kikusui sakusen ) was a series of suicidal air attacks by Imperial Japanese forces during the Battle of Okinawa against Allied fleets
in the waters around Okinawa, as part of Operation Ten-Go.

 The name of the operation, "Kikusui", comes from the hata-jirushi of the samurai Kusunoki Masashige.
</item 5>

<item 6>
Operation Diadem
Operation Diadem, also referred to as the Fourth Battle of Monte Cassino or, in Canada, the Battle of the Liri Valley, was an offensive operation undertaken by
the Allies of World War II (U.S. Fifth Army and British Eighth Army in May 1944, as part of the Italian Campaign of World War II.

 "Diadem" was supported by air attacks called Operation Strangle.

 The opposing force was the German 10th Army.
</item 6>

<item 7>
Operation Epsom
Operation Epsom, also known as the First Battle of the Odon, was a British Second World War offensive that took place between 26 and 30 June 1944, during the
Battle of Normandy.

 The offensive was intended to outflank and seize the German-occupied city of Caen, an important Allied objective, in the early stages of Operation Overlord,
the Allied invasion of north-west Europe.
</item 7>

<item 8>
Battle of Attu
The Battle of Attu, which took place from 11–30 May 1943, was a battle fought between forces of the United States, aided by Canadian reconnaissance and
fighter-bomber support, and the Empire of Japan on Attu Island off the coast of the Territory of Alaska as part of the Aleutian Islands Campaign during the
American Theater and the Pacific Theater and was the only land battle of World War II fought on incorporated territory of the United States.

 It is also the only land battle in which Japanese and American forces fought in Arctic conditions.
</item 8>

<item 9>
Battle of Okinawa
The Battle of Okinawa (Japanese: 沖縄戦 , Hepburn: Okinawa-sen ) (Okinawan: 沖縄戦 , "Uchinaa ikusa " ), codenamed Operation Iceberg, was a major battle of the
Pacific War fought on the island of Okinawa by United States Marine and Army forces against the Imperial Japanese Army and included the largest amphibious
assault in the Pacific War during World War II, the April 1, 1945, invasion of Okinawa itself.

 The 82-day battle lasted from April 1 until June 22, 1945.

 After a long campaign of island hopping, the Allies were planning to use Okinawa, a large island only 340 mi away from mainland Japan, as a base for air
operations for the planned invasion of the Japanese home islands.
</item 9>

<item 10>
Operation Diadem order of battle
Operation Diadem order of battle is a listing of the significant formations that were involved in the fighting on the Winter Line and at the Anzio bridgehead
south of Rome during "Operation Diadem" in May - June 1944 which resulted in the Allied breakthrough at Cassino and the breakout at Anzio leading to the capture
of Rome.
</item 10>
=== END ITEM id=5ee90058fdf8d1f4 ===

=== ITEM id=6c806f8f9c4f6b48 ===
Gold sentences and facts:
Sentence 0: The Doris Bither case, also known as the Entity hauntings, was an alleged haunting which occurred in 1974 at Culver City, California where a woman named Doris Bither alleged the ghosts of three Asian men were raping her.
  Fact 0.0: The Doris Bither case is also known as the Entity hauntings.
  Fact 0.1: The Doris Bither case was an alleged haunting.
  Fact 0.2: The alleged haunting occurred in 1974.
  Fact 0.3: The alleged haunting occurred at Culver City, California.
  Fact 0.4: Doris Bither alleged that the ghosts of three Asian men were raping her.
Sentence 1: The Entity is a 1982 American horror film directed by Sidney J. Furie and written by Frank De Felitta, who adapted his 1978 novel of the same name.
  Fact 1.0: The Entity is a 1982 American horror film.
  Fact 1.1: The Entity was directed by Sidney J. Furie.
  Fact 1.2: The Entity was written by Frank De Felitta.
  Fact 1.3: Frank De Felitta adapted his 1978 novel of the same name for The Entity.
Sentence 2:  Like the novel, the film is based on the 1974 Doris Bither case.
  Fact 2.0: The film is based on the 1974 Doris Bither case.
  Fact 2.1: The novel is also based on the 1974 Doris Bither case.

Retrieved items:
<item 1>
Night Life (film)
Night Life is a 1989 American zombie comedy film directed by David Acomba and starring Scott Grimes, John Astin, and Cheryl Pollak.

 It centers around a high school nerd named Archie Melville who is harassed from beyond the grave by the zombie corpses of his former high school bullies
(recently deceased) after they are brought back to life by a freak lightning storm.

 It is also known as Grave Misdemeanors.
</item 1>

<item 2>
Hardcore (1979 film)
Hardcore is a 1979 American crime drama film written and directed by Paul Schrader and starring George C. Scott, Peter Boyle and Season Hubley.

 The story concerns a father searching for his daughter, who has vanished only to appear in a pornographic film.

 Writer-director Schrader had previously written the screenplay for Martin Scorsese's "Taxi Driver", and both films share a theme of exploring an unseen
subculture.
</item 2>

<item 3>
Midnight (1934 film)
Midnight is a 1934 American drama film directed by Chester Erskine and starring Sidney Fox, O.P. Heggie, Henry Hull and Margaret Wycherly.

 The film was produced for Universal and was shot on a modest budget of $50,000 at Thomas Edison Studios, which producer/director Chester Erskine had re-opened
specifically for the shoot.
</item 3>

<item 4>
Doris Bither case
The Doris Bither case, also known as the Entity hauntings, was an alleged haunting which occurred in 1974 at Culver City, California where a woman named Doris
Bither alleged the ghosts of three Asian men were raping her.

 The case inspired Frank De Felitta's 1978 book "The Entity" which was made into a 1982 film of the same name starring Barbara Hershey.
</item 4>

<item 5>
Little Fauss and Big Halsy
Little Fauss and Big Halsy is a 1970 film directed by Sidney J. Furie, starring Robert Redford and Michael J. Pollard, also featuring Lauren Hutton, Noah Beery,
Jr. and Lucille Benson.
</item 5>

<item 6>
The Entity
The Entity is a 1982 American horror film directed by Sidney J. Furie and written by Frank De Felitta, who adapted his 1978 novel of the same name.

 It stars Barbara Hershey as a woman who is raped and tormented by an invisible assailant.

 Despite being filmed and planned for a release in 1981, the movie was not released in worldwide theaters until September 1982 and February 1983 in the United
States.

 Like the novel, the film is based on the 1974 Doris Bither case.
</item 6>

<item 7>
Sheila Levine Is Dead and Living in New York
Sheila Levine Is Dead and Living in New York is a 1975 film directed by Sidney J. Furie, about a shy young woman who moves to New York City and falls in love
with the boyfriend of her extroverted roommate.

 The film was co-written by Kenny Solms and Gail Parent, and based on her novel.

 The film was shot on location in New York City.
</item 7>

<item 8>
Doctor Blood's Coffin
Doctor Blood's Coffin is a 1961 British horror film directed by Sidney J. Furie.

 Kieron Moore stars as a mad scientist who reanimates the dead.
</item 8>

<item 9>
The Rocky Horror Picture Show
The Rocky Horror Picture Show is a 1975 British-American musical screwball comedy horror film by 20th Century Fox produced by Lou Adler and Michael White and
directed by Jim Sharman.

 The screenplay was written by Sharman and actor Richard O'Brien, who appears in the film, which is based on the 1973 musical stage production "of the same
title", with music, book, and lyrics by O'Brien.

 The production is a parody tribute to the science fiction and horror B movies of the 1930s through to the early 1970s.

 Along with O'Brien, the film stars Tim Curry, Susan Sarandon, and Barry Bostwick and is narrated by Charles Gray with cast members from the original Royal
Court Theatre, Roxy Theatre, and Belasco Theatre productions.
</item 9>

<item 10>
The Snake Woman
The Snake Woman is a low budget 1961 British horror film directed by Sidney J. Furie and starring Susan Travers, John McCarthy.
</item 10>
=== END ITEM id=6c806f8f9c4f6b48 ===

=== ITEM id=e2e9fcf01f320522 ===
Gold sentences and facts:
Sentence 0: Anaïs Nin (] ; born Angela Anaïs Juana Antolina Rosa Edelmira Nin y Culmell; February 21, 1903 – January 14, 1977) was an essayist and memoirist born to Cuban parents in France, where she was also raised.
  Fact 0.0: Anaïs Nin was born Angela Anaïs Juana Antolina Rosa Edelmira Nin y Culmell.
  Fact 0.1: Anaïs Nin was born on February 21, 1903.
  Fact 0.2: Anaïs Nin died on January 14, 1977.
  Fact 0.3: Anaïs Nin was an essayist.
  Fact 0.4: Anaïs Nin was a memoirist.
  Fact 0.5: Anaïs Nin was born to Cuban parents.
  Fact 0.6: Anaïs Nin was born in France.
  Fact 0.7: Anaïs Nin was raised in France.
Sentence 1: Odysseas Elytis (Greek: Οδυσσέας Ελύτης , ] , pen name of Odysseas Alepoudellis, Greek: Οδυσσέας Αλεπουδέλλης ; 2 November 1911 – 18 March 1996) was regarded as a major exponent of romantic modernism in Greece and the world.
  Fact 1.0: Odysseas Elytis is a pen name of Odysseas Alepoudellis.
  Fact 1.1: Odysseas Elytis's Greek name is Οδυσσέας Ελύτης.
  Fact 1.2: Odysseas Alepoudellis's Greek name is Οδυσσέας Αλεπουδέλλης.
  Fact 1.3: Odysseas Elytis was born on 2 November 1911.
  Fact 1.4: Odysseas Elytis died on 18 March 1996.
  Fact 1.5: Odysseas Elytis was regarded as a major exponent of romantic modernism in Greece.
  Fact 1.6: Odysseas Elytis was regarded as a major exponent of romantic modernism in the world.
Sentence 2:  In 1979 he was awarded the Nobel Prize in Literature.
  Fact 2.0: Odysseas Elytis was awarded the Nobel Prize in Literature in 1979.

Retrieved items:
<item 1>
Anaïs Nin
Anaïs Nin (] ; born Angela Anaïs Juana Antolina Rosa Edelmira Nin y Culmell; February 21, 1903 – January 14, 1977) was an essayist and memoirist born to Cuban
parents in France, where she was also raised.

 She spent some time in Spain and Cuba, but lived most of her life in the United States, where she became an established author.

 She wrote journals (which span more than 60 years, beginning when she was 11 years old and ending shortly before her death), novels, critical studies, essays,
short stories, and erotica.

 A great deal of her work, including "Delta of Venus" and "Little Birds", was published posthumously.
</item 1>

<item 2>
Henry and June
Henry and June: From the Unexpurgated Diary of Anaïs Nin (full title "Henry and June: From A Journal of Love: the Unexpurgated Diary of Anaïs Nin (1931–1932)")
is a 1986 book that is based upon material excerpted from the unpublished diaries of Anaïs Nin.
</item 2>

<item 3>
Incest: From a Journal of Love
Incest: From a Journal of Love: The Unexpurgated Diary of Anaïs Nin (1932–1934) is a 1992 non-fiction book by Anaïs Nin.

 It is a continuation of the diary entries first published in "Henry and June: From the Unexpurgated Diary of Anaïs Nin".

 It features Nin's relationships with writer Henry Miller, his wife June Miller, the psychoanalyst Otto Rank, her father Joaquín Nin, and her husband Hugh
Parker Guiler.

 She also copied some of her correspondence with these people into her diary.

 Much of this book was written in English, although those of her letters which were originally written in French and Spanish were translated.

 Most of this diary takes place in France, particularly Clichy, Paris and Louveciennes.
</item 3>

<item 4>
Seduction of the Minotaur
Seduction of the Minotaur is an autobiographical novel by the mixed nationality writer Anaïs Nin, the last part of her "Cities of the Interior" sequence.

 It is about a woman named Lillian, and her self-psychoanalysis.

 The setting is taken from Anaïs' diary account of her first trip to Acapulco in 1947, and the novel repeats much of the first part of "The Diary of Anaïs Nin"
volume V.

 Since the author was concerned with psychology rather than physical adventure, there is actually less violence in the novel than in the diary account.

 The exception is that the doctor allows himself to be shot because he is loved only as a doctor and never as a man, perhaps patterned after her understanding
of Otto Rank's death.
</item 4>

<item 5>
Odysseas Elytis
Odysseas Elytis (Greek: Οδυσσέας Ελύτης , ] , pen name of Odysseas Alepoudellis, Greek: Οδυσσέας Αλεπουδέλλης ; 2 November 1911 – 18 March 1996) was regarded as
a major exponent of romantic modernism in Greece and the world.

 In 1979 he was awarded the Nobel Prize in Literature.
</item 5>

<item 6>
Nobel Prize in Literature
Since 1901, the Nobel Prize in Literature (Swedish: "Nobelpriset i litteratur" ) has been awarded annually to an author from any country who has, in the words
of the will of Alfred Nobel, produced "in the field of literature the most outstanding work in an ideal direction" (original Swedish: "den som inom litteraturen
har producerat det mest framstående verket i en idealisk riktning").

 Though individual works are sometimes cited as being particularly noteworthy, here "work" refers to an author's work as a whole.

 The Swedish Academy decides who, if anyone, will receive the prize in any given year.

 The academy announces the name of the chosen laureate in early October.

 It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895; the others are the Nobel Prize in Chemistry, Nobel Prize in Physics, Nobel
Peace Prize, and Nobel Prize in Physiology or Medicine.
</item 6>

<item 7>
Anaïs Nin: A Biography
Anaïs Nin: A Biography is Deirdre Bair's award-winning biography of writer Anaïs Nin.

 It is considered arguably by many to be the most comprehensive, well-researched, and scholarly biography available of Nin.

 Though the biography has received praise, it has also angered some fans of Anaïs Nin as well as some of her former associates, some of whom claim that Bair's
critical and rigorous investigation of Anaïs Nin's life is unsympathetic.
</item 7>

<item 8>
The Diary of Anaïs Nin
The Diary of Anaïs Nin is the published version of Anaïs Nin's own private manuscript diary, which she began at age 11 in 1914 during a trip from Europe to New
York with her mother and two brothers.

 Anaïs Nin would later say she had begun the diary as a letter to her father, Cuban composer Joaquín Nin, who had abandoned the family a few years earlier.
</item 8>

<item 9>
Nearer the Moon
Nearer the Moon: From a Journal of Love (full title "Nearer the Moon: From a Journal of Love, the Unexpurgated Diary of Anaïs Nin (1937–1939)") is a 1996 book
based upon material excerpted from the unpublished diaries of Anaïs Nin.

 It corresponds temporally to part of Nin's published diaries.

 It consists mainly of material that was left out of the published version because it would have hurt people involved or their relationships with Anaïs Nin had
it been published at the time.
</item 9>

<item 10>
Collages (novel)
Collages, published in 1964, was Anaïs Nin's last published novel (excluding her erotica).

 It is very different from the previous novels of the "Cities of the Interior" series, because it contains none of the familiar characters in those novels.

 It is also different in that it takes place on two continents, has about two dozen important characters, and for the first time in Anaïs Nin's work the female
lead is not seeking psychological wholeness.

 Most of Anaïs Nin's other novels only have a few main characters, there is rarely much geographical movement in the other novels, and usually the lead
character is a woman seeking psychological wholeness.
</item 10>
=== END ITEM id=e2e9fcf01f320522 ===

=== ITEM id=07bc168e238ea021 ===
Gold sentences and facts:
Sentence 0: Edward Sharpe and the Magnetic Zeros is an American folk rock band formed in Los Angeles, California in 2007.
  Fact 0.0: Edward Sharpe and the Magnetic Zeros is an American folk rock band.
  Fact 0.1: Edward Sharpe and the Magnetic Zeros was formed in Los Angeles, California.
  Fact 0.2: Edward Sharpe and the Magnetic Zeros was formed in 2007.
Sentence 1: Louis XIV is an American rock band from San Diego, California.
  Fact 1.0: Louis XIV is an American rock band.
  Fact 1.1: Louis XIV is from San Diego, California.
Sentence 2:  The band has released four EPs between 2003 and 2007, and three albums between 2003 and 2008, the latter two of which were distributed by Atlantic Records.
  Fact 2.0: The band released four EPs between 2003 and 2007.
  Fact 2.1: The band released three albums between 2003 and 2008.
  Fact 2.2: Two of the band's three albums, the latter two, were distributed by Atlantic Records.

Retrieved items:
<item 1>
Louis XIV (band)
Louis XIV is an American rock band from San Diego, California.

 The band has released four EPs between 2003 and 2007, and three albums between 2003 and 2008, the latter two of which were distributed by Atlantic Records.

 The band broke up in 2009, but in 2013, in an interview with The Reno Dispatch, Jason Hill confirmed that the band had decided to reunite.
</item 1>

<item 2>
PersonA
PersonA (pronounced "Persona") is the fourth studio album by the band Edward Sharpe and the Magnetic Zeros.

 It was released on April 15, 2016.
</item 2>

<item 3>
Edward Sharpe and the Magnetic Zeros (album)
Edward Sharpe and the Magnetic Zeros is the third album by Edward Sharpe and the Magnetic Zeros.

 It was released on July 23, 2013 in North America and was released on July 29, 2013 around the world through Vagrant Records, Rough Trade Records and Communion
Records.

 Frontman Alex Ebert stated that "These songs mean everything to me - It's the rawest, most liberated, most rambunctious stuff we've done."
</item 3>

<item 4>
Manimal Vinyl
Manimal Vinyl is a Calabasas based record label founded in 2006 by experimental musician and fashion stylist, Paul Beahan.

 They were originally known as a psychedelic folk label that makes semi-annual tribute records for worldwide charities, having eventually switched to dance
music and progressive underground.

 They are the label who has release debut records from act ranging from Warpaint to Bat for Lashes and exclusive singles from Moby, Duran Duran, Carla Bruni,
Edward Sharpe and the Magnetic Zeros.

 In 2008, they founded the annual Manimal Festival in Pioneertown, California which ran from 2008-2010.

 The festival has featured line-ups with Edward Sharpe and the Magnetic Zeros, Warpaint, Ariel Pink and papercranes.

 Manimal has offices in Los Angeles and New York City.
</item 4>

<item 5>
Edward Sharpe and the Magnetic Zeros
Edward Sharpe and the Magnetic Zeros is an American folk rock band formed in Los Angeles, California in 2007.

 The group is led by lead singer Alex Ebert.

 The band's name is based on a story Ebert wrote in his youth, about a messianic figure named Edward Sharpe.

 Drawing from roots rock, folk, gospel, and psychedelic music, the band's image and sound evoke the hippie movement of the 1960s and 1970s.

 The group's first show was played July 18, 2007 at The Troubadour in West Hollywood, California.

 Their first studio album, "Up from Below", was released on July 7, 2009 on Community Records and featured the popular single "Home".

 The group released their second full-length album, "Here", on May 29, 2012, and third album, "Edward Sharpe and the Magnetic Zeros", on July 23, 2013.

 Their fourth studio album, "PersonA", was released in April 2016.
</item 5>

<item 6>
40 Day Dream/Geez Louise
"40 Day Dream"/"Geez Louise" is a 7" vinyl single released in the UK to succeed the album "Up from Below" by Edward Sharpe and the Magnetic Zeros.

 It appeared on the sixth episode of the third season of "Chuck" (entitled "Chuck Versus the Nacho Sampler" and airing on February 1, 2010); Episode 2 of HBO's
"Hung"; "Jimmy Kimmel Live!

" on November 11, 2010; and "Conan" on December 15, 2010.
</item 6>

<item 7>
Alexander (Alex Ebert album)
Alexander is the first solo album by American musician Alex Ebert, lead singer of Edward Sharpe and the Magnetic Zeros and Ima Robot.

 It was released on March 1, 2011 under the Community Records label.

 The song "Truth" was featured in the season 4 premiere of AMC's "Breaking Bad" titled "Box Cutter".
</item 7>

<item 8>
The White Stripes
The White Stripes were an American rock duo formed in 1997 in Detroit, Michigan.

 The group consisted of Jack White (songwriter, vocals, guitar, piano, and mandolin) and Meg White (drums and vocals).

 After releasing several singles and three albums within the Detroit music scene, The White Stripes rose to prominence in 2002, as part of the garage rock
revival scene.

 Their successful and critically acclaimed albums "White Blood Cells" and "Elephant" drew attention from a large variety of media outlets in the United States
and the United Kingdom, with the single "Seven Nation Army" and its guitar line becoming their signature song.

 The band recorded two more albums, "Get Behind Me Satan" in 2005 and "Icky Thump" in 2007, and dissolved in 2011 after a lengthy hiatus from performing and
recording.
</item 8>

<item 9>
Orpheo McCord
Orpheo McCord (born 1979) is an American drummer and percussionist.

 He is an original member of Edward Sharpe and the Magnetic Zeros, and a former member of The Fall, Fool's Gold and Cass McCombs.
</item 9>

<item 10>
Lingdian
Zero Band () are a Chinese rock band from Inner Mongolia.

 The lead singer is Zhou Xiaoou (周晓鸥).

 The English name "Zero Band" was used in Beijing concert advertisements, though the band did not originally use an English name; the name has also been
translated as "Point Zero" in some older English publications.

 The band appeared on the Jingwen label in China, but were licensed outside China by JVC.

 One of the top Beijing bands at the end of the 1990s, Lingdian originally suffered the same resistance from government-owned venues as other indigenous PRC
rock bands.
</item 10>
=== END ITEM id=07bc168e238ea021 ===

=== ITEM id=60bc3a747d86e9c2 ===
Gold sentences and facts:
Sentence 0: Martin Degville is (born 27 January 1961 in Walsall) is the lead singer and co-songwriter of the UK pop band, Sigue Sigue Sputnik – which had a worldwide hit single in 1986 with "Love Missile F1-11" – and six other EMI single releases.
  Fact 0.0: Martin Degville was born 27 January 1961 in Walsall.
  Fact 0.1: Martin Degville is the lead singer of the UK pop band Sigue Sigue Sputnik.
  Fact 0.2: Martin Degville is a co-songwriter of Sigue Sigue Sputnik.
  Fact 0.3: Sigue Sigue Sputnik had a worldwide hit single in 1986 with 'Love Missile F1-11'.
  Fact 0.4: Sigue Sigue Sputnik has six other EMI single releases.
Sentence 1: Paul Richard "Richie" Furay (born May 9, 1944) is an American singer, songwriter, and Rock & Roll Hall of Fame member who is best known for forming the bands Buffalo Springfield with Stephen Stills, Neil Young, Bruce Palmer, and Dewey Martin, and Poco with Jim Messina, Rusty Young, George Grantham and Randy Meisner.
  Fact 1.0: Paul Richard 'Richie' Furay was born May 9, 1944.
  Fact 1.1: Richie Furay is an American singer.
  Fact 1.2: Richie Furay is an American songwriter.
  Fact 1.3: Richie Furay is a Rock & Roll Hall of Fame member.
  Fact 1.4: Richie Furay is best known for forming Buffalo Springfield with Stephen Stills, Neil Young, Bruce Palmer, and Dewey Martin.
  Fact 1.5: Richie Furay is best known for forming Poco with Jim Messina, Rusty Young, George Grantham, and Randy Meisner.

Retrieved items:
<item 1>
Poco: The Songs of Richie Furay
Poco: The Songs of Richie Furay is a compilation album consisting of songs by Richie Furay during his tenure with the band Poco, released in 1980.
</item 1>

<item 2>
Jerry Garcia
Jerome John Garcia (August 1, 1942 – August 9, 1995) was an American singer-songwriter and guitarist, best known for his work as the lead guitarist and as a
vocalist with the band the Grateful Dead, which came to prominence during the counterculture era in the 1960s.

 Although he disavowed the role, Garcia was viewed by many as the leader or "spokesman" of the group.
</item 2>

<item 3>
Richie Furay
Paul Richard "Richie" Furay (born May 9, 1944) is an American singer, songwriter, and Rock & Roll Hall of Fame member who is best known for forming the bands
Buffalo Springfield with Stephen Stills, Neil Young, Bruce Palmer, and Dewey Martin, and Poco with Jim Messina, Rusty Young, George Grantham and Randy Meisner.

 His best known song (originally written during his tenure in Buffalo Springfield, but eventually performed by Poco, as well) was "Kind Woman", which he wrote
for his wife, Nancy.
</item 3>

<item 4>
Sunshine Becker
Sunshine Becker (born Sunshine Flower Garcia on July 1, 1972) is an American singer who performed backing vocals for the band Furthur.

 Despite her maiden name, Garcia, she is not related to Jerry Garcia, an incorrect assumption made by some because of her involvement with Furthur, a
post-Garcia incarnation of the Grateful Dead.

 Similarly, despite her first name, Sunshine, she is not to be confused with Sunshine Kesey, daughter of Ken Kesey and Carolyn Adams (aka Mountain Girl or MG),
Jerry Garcia's second wife.
</item 4>

<item 5>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 5>

<item 6>
Dean Martin
Dean Martin (born Dino Paul Crocetti; June 7, 1917 – December 25, 1995) was an American singer, actor, comedian, and film producer.

 One of the most popular and enduring American entertainers of the mid-20th century, Martin was nicknamed the "King of Cool" for his seemingly effortless
charisma and self-assurance.
</item 6>

<item 7>
Martin Degville
Martin Degville is (born 27 January 1961 in Walsall) is the lead singer and co-songwriter of the UK pop band, Sigue Sigue Sputnik – which had a worldwide hit
single in 1986 with "Love Missile F1-11" – and six other EMI single releases.

 Sputnik was formed with ex-Generation X bassist Tony James.
</item 7>

<item 8>
Dean Martin Sings
Dean Martin Sings is the first studio album by Italian-American singer Dean Martin, released in 1953.

 It is the first long-play 10-inch album recorded by Martin for Capitol Records during two sessions recorded on the evening of November 20, 1952.

 The first session was recorded between 5 and 8 PM and it produced five songs featuring string arrangements.

 "There's My Lover" was recorded but not released.

 After a ninety-minute break, Martin was joined by a brass arrangement to record the remaining four songs.

 Seven of the eight songs on this album appeared in the Martin & Lewis film, "The Stooge".

 Two years later, the songs from this 10-inch album would be combined with four newly popular songs recorded between 1951 and 1953 to create a full-length
12-inch album.

 The 2005 Collectors' Choice reissue added four bonus songs recorded between 1949 and 1953 and was released with alternative cover artwork.
</item 8>

<item 9>
Souther–Hillman–Furay Band
The Souther Hillman Furay Band (SHF) was a country rock supergroup led by singer-songwriters Richie Furay (Buffalo Springfield, Poco), Chris Hillman (The Byrds,
The Flying Burrito Brothers, Manassas), and J.D. Souther (Longbranch Pennywhistle, noted songwriter for Linda Ronstadt and the Eagles).

 The band recorded two albums during the 1970s before breaking up due to disagreements between the members.
</item 9>

<item 10>
Martin Flavin
Martin Archer Flavin (November 2, 1883 – December 27, 1967) was an American playwright and novelist.
</item 10>
=== END ITEM id=60bc3a747d86e9c2 ===

=== ITEM id=1ec3a90468813680 ===
Gold sentences and facts:
Sentence 0:  They were first year members of the Mountain West Conference.
  Fact 0.0: They were first-year members of the Mountain West Conference.
Sentence 1: The Mountain West Conference (MW) is one of the collegiate athletic conferences affiliated with the National Collegiate Athletic Association (NCAA) Division I Football Bowl Subdivision (FBS) (formerly I-A).
  Fact 1.0: The Mountain West Conference (MW) is a collegiate athletic conference.
  Fact 1.1: The Mountain West Conference is one of the collegiate athletic conferences affiliated with the National Collegiate Athletic Association (NCAA).
  Fact 1.2: The Mountain West Conference is affiliated with the NCAA Division I Football Bowl Subdivision (FBS).
  Fact 1.3: The Football Bowl Subdivision (FBS) was formerly called I-A.

Retrieved items:
<item 1>
Men's National Collegiate Athletic Association
The Men's National Collegiate Athletic Association is an athletic association in the Philippines, and is the direct counterpart of the older Women's National
Collegiate Athletic Association (founded in 1970).

 The Men's National Collegiate Athletic Association (MNCAA) is exclusive for men.

 The MNCAA was founded in 2004.

 Competition is divided into three divisions: Seniors for college students, Juniors for high school students, and Midgets for grade school and first year high
school students.

 The competing sports of the tournament are Basketball, Volleyball, Futsal, Table Tennis, Badminton and Streetdance.
</item 1>

<item 2>
2011–12 Wagner Seahawks men's basketball team
The 2011–12 Wagner Seahawks men's basketball team represented Wagner College during the 2011–12 NCAA Division I men's basketball season.

 The Seahawks, led by second year head coach Dan Hurley, played their home games at Spiro Sports Center and are members of the Northeast Conference.

 They finished the season 25–6, 15–3 in NEC play to finish in second place.

 They lost in the semifinals of the NEC Basketball Tournament to Robert Morris.

 Despite having 25 wins, the Seahawks did not accept an invitation to a post season tournament.
</item 2>

<item 3>
2013–14 Iona Gaels men's basketball team
The 2013–14 Iona Gaels men's basketball team represented Iona College during the 2013–14 NCAA Division I men's basketball season.

 The Gaels, led by fourth year head coach Tim Cluess, played their home games at the Hynes Athletic Center and were members of the Metro Atlantic Athletic
Conference.

 They finished the season 22–11, 17–3 to win the MAAC regular season championship.

 They advanced to the championship game of the MAAC Tournament where they lost to Manhattan.

 As a regular season conference champion who failed to win their conference tournament, they received an automatic bid to the National Invitation Tournament
where they lost in the first round to Louisiana Tech.
</item 3>

<item 4>
2012 Hawaii Warriors football team
The 2012 Hawaii Warriors football team represented the University of Hawaii at Manoa in the 2012 NCAA Division I FBS football season.

 They were led by first-year head coach Norm Chow and played their home games at Aloha Stadium.

 They were first year members of the Mountain West Conference.

 They finished the season 3–9, 1–7 in Mountain West play to finish in a tie for ninth place.
</item 4>

<item 5>
Hawaii Rainbow Warriors baseball
The Hawaii Rainbow Warriors baseball team is a varsity intercollegiate athletic team of the University of Hawaii at Manoa in Honolulu, Hawaii, United States.

 The team is a member of the Big West Conference, which is part of the National Collegiate Athletic Association's Division I.

 Hawaii's first baseball team was fielded in 1923.

 The team plays its home games at Les Murakami Stadium in Honolulu, Hawaii.

 The Rainbow Warriors are coached by Mike Trapasso.
</item 5>

<item 6>
List of Pacific Tigers head football coaches
The Pacific Tigers college football team represented University of the Pacific (CA).

 The Tigers competed in the National Collegiate Athletic Association (NCAA) College Division in the years 1937–1968.

 In 1969, the team moved to National Collegiate Athletic Association (NCAA) Division I and stayed there until disbanding after the 1995 season.
</item 6>

<item 7>
Wayne State Warriors football
The Wayne State Warriors football team is the college football team at Wayne State University in Detroit, Michigan.

 The Wayne State football team played their first game in October 1918.

 The Wayne State Warriors have competed in the Great Lakes Intercollegiate Athletic Conference since 1999 (and previously from 1975-1989), and are currently a
Division II member of the National Collegiate Athletic Association (NCAA).

 Wayne State plays their home games at Tom Adams Field at Wayne State Stadium.

 All Wayne State games are broadcast on WDTK radio.
</item 7>

<item 8>
Women's National Collegiate Athletic Association
The Women's National Collegiate Athletic Association (WNCAA) is an athletic association in the Philippines exclusively for women.

 It was founded in 1970.

 Competition is divided into three divisions: Seniors for college students, Juniors for high school students, and Midgets for grade school and first year high
school students.

 Its men's counterpart is the Men's National Collegiate Athletic Association, founded in 2004.
</item 8>

<item 9>
Mountain West Conference
The Mountain West Conference (MW) is one of the collegiate athletic conferences affiliated with the National Collegiate Athletic Association (NCAA) Division I
Football Bowl Subdivision (FBS) (formerly I-A).

 The MW officially began operations in July 1999.

 Geographically, the MW covers a broad expanse of the Western United States, with member schools located in California, Colorado, Hawaii, Idaho, Nevada, New
Mexico, Utah, and Wyoming.

 Craig Thompson has served as Commissioner of the MW since its founding in 1999.
</item 9>

<item 10>
2012–13 Wagner Seahawks men's basketball team
The 2012–13 Wagner Seahawks men's basketball team represented Wagner College during the 2012–13 NCAA Division I men's basketball season.

 The Seahawks were led by the youngest men's head coach in NCAA Division I, Bashir Mason, who was 28 when he was elevated from an assistant position in March
2012 following the departure of Dan Hurley for Rhode Island.

 The Seahawks played their home games at Spiro Sports Center and were members of the Northeast Conference.

 They finished the season 19–12, 12–6 in NEC play to finish in a three way tie for second place.

 They advanced to the semifinals of the Northeast Tournament where they lost to Long Island.

 For the second consecutive year, despite a winning record, Wagner choose not to participate in a post season tournament.
</item 10>
=== END ITEM id=1ec3a90468813680 ===

=== ITEM id=4214a0ab30967d82 ===
Gold sentences and facts:
Sentence 0: Saddle Rock Elementary School or simply "Saddle Rock" is an elementary school, comprising grades Kindergarten through 5.
  Fact 0.0: Saddle Rock Elementary School is also called Saddle Rock.
  Fact 0.1: Saddle Rock Elementary School is an elementary school.
  Fact 0.2: Saddle Rock Elementary School comprises grades Kindergarten through 5.
Sentence 1:  It is a public school located in Great Neck, New York, USA as part of the Great Neck School District.
  Fact 1.0: Saddle Rock Elementary School is a public school.
  Fact 1.1: Saddle Rock Elementary School is located in Great Neck, New York, USA.
  Fact 1.2: Saddle Rock Elementary School is part of the Great Neck School District.
Sentence 2: The Great Neck School District is a comprehensive community public school district primarily serving students in Great Neck, New York.
  Fact 2.0: The Great Neck School District is a comprehensive community public school district.
  Fact 2.1: The Great Neck School District primarily serves students in Great Neck, New York.
Sentence 3:  It is Union Free School District Number 7 in the Town of North Hempstead, Nassau County, New York, United States, on Long Island.
  Fact 3.0: The Great Neck School District is Union Free School District Number 7.
  Fact 3.1: The Great Neck School District is in the Town of North Hempstead.
  Fact 3.2: The Great Neck School District is in Nassau County, New York, United States.
  Fact 3.3: The Great Neck School District is on Long Island.

Retrieved items:
<item 1>
New York State Route 108
New York State Route 108 (NY 108) is a 1.72 mi north–south state highway located on the Suffolk County side of the Suffolk–Nassau county line on Long Island,
New York, in the United States.

 It is a spur route connecting NY 25A in Cold Spring Harbor to the Cold Spring Harbor station on the Long Island Rail Road's Port Jefferson Branch via Harbor
Road.

 Harbor Road terminates at an intersection with Woodbury Road, on the Nassau County line, which carries County Route 11 to the east and unsigned County Route 12
to the west.

 NY 108, assigned in the early 1930s, is the shortest state highway on Long Island.
</item 1>

<item 2>
Great Neck, New York
Great Neck is a region on Long Island, New York, that covers a peninsula on the North Shore of Long Island, which includes the villages of Great Neck, Great
Neck Estates, Great Neck Plaza, and others, as well as an area south of the peninsula near Lake Success and the border territory of Queens.

 The incorporated village of Great Neck had a population of 9,989 at the 2010 census, while the larger Great Neck area comprises a residential community of some
40,000 people in nine villages and hamlets in the town of North Hempstead, of which Great Neck is the northwestern quadrant.

 Great Neck has five ZIP Codes (11020–11024), and one school district.
</item 2>

<item 3>
East Lyme High School
East Lyme High School is a high school located in the Flanders Village region of East Lyme, Connecticut.

 It is operated by East Lyme Public Schools.

 The mascot is Sven the Viking.

 Students from the town of Salem, Connecticut in grades 9 through 12 attend high school in East Lyme (as they have no high school of their own); this will be
the case until at least 2016 when the current co-op agreement between the two towns expires.
</item 3>

<item 4>
Saddle Rock Elementary School
Saddle Rock Elementary School or simply "Saddle Rock" is an elementary school, comprising grades Kindergarten through 5.

 It is a public school located in Great Neck, New York, USA as part of the Great Neck School District.

 The school has been recognized by the Blue Ribbon Schools Program.
</item 4>

<item 5>
Old Country Road
Old Country Road is a major east–west thoroughfare through central Nassau County and extending into western Suffolk County on Long Island, New York.

 It serves many of the major shopping centers in central Nassau County including Roosevelt Field Mall.

 The road also forms part of the border between the towns of Hempstead and North Hempstead.

 In Nassau County, Old Country Road is county-maintained as the unsigned County Route 25 (CR 25) west of Round Swamp Road.
</item 5>

<item 6>
St. Mary's High School (Manhasset, New York)
Saint Mary's High School is a private Catholic high school in Manhasset, New York.

 St. Mary's parish is part of the Roman Catholic Diocese of Rockville Centre yet the High School serves families of the Roman Catholic Archdiocese of New York,
the Diocese of Brooklyn and the Diocese of Rockville Centre.

 It is located off Northern Boulevard at 51 Clapham Avenue, Manhasset, near the Miracle Mile.

 St. Mary's Men's Varsity Ice Hockey is the all-time leader in State Championships.
</item 6>

<item 7>
Great Neck School District
The Great Neck School District is a comprehensive community public school district primarily serving students in Great Neck, New York.

 It also serves students in some parts of Manhasset and New Hyde Park.

 It is Union Free School District Number 7 in the Town of North Hempstead, Nassau County, New York, United States, on Long Island.
</item 7>

<item 8>
Village School (Great Neck, New York)
Village School (also known as VS) is an American long-established public alternative school.

 It is a member of Coalition of Essential Schools and is located in the village of Great Neck, New York, serving students in grades 8 through 12.

 Village School is one of three high schools in the Great Neck School District, which includes Great Neck North High School and Great Neck South High School.

 Village School offers its 39 students an outdoor education program, college preparatory program, and inclusion of students with disabilities.
</item 8>

<item 9>
New Jersey Japanese School
The New Jersey Japanese School (ニュージャージー日本人学校 , "Nyūjājī Nihonjin Gakkō" , NJJS) is a Japanese school located in Oakland, New Jersey, United States in the New
York City metropolitan area.

 It is one of the two Japanese day schools operated by the Japanese Educational Institute of New York (JEI; ニューヨーク日本人教育審議会 "Nyūyōku Nihonjin Kyōiku Shingi
Kai"), a nonprofit organization which also operates two Japanese weekend school systems in the New York City area.
</item 9>

<item 10>
New Rochelle Harbor (Long Island Sound)
New Rochelle Harbor is the name of a harbor located along Long Island Sound in the city of New Rochelle in Westchester County, New York.

 The Davenport Neck peninsula off the mainland divides New Rochelle's waterfront into two bays; the westerly referred to as New Rochelle Harbor and the easterly
as Echo Bay.
</item 10>
=== END ITEM id=4214a0ab30967d82 ===

=== ITEM id=edb0095ea1825935 ===
Gold sentences and facts:
Sentence 0:  It was driven by Eddie Irvine, who was in his second year with the team, and Luciano Burti, who had deputised for Irvine at the 2000 Austrian Grand Prix.
  Fact 0.0: The car described in this sentence was driven by Eddie Irvine.
  Fact 0.1: Eddie Irvine was in his second year with the team.
  Fact 0.2: The car described in this sentence was also driven by Luciano Burti.
  Fact 0.3: Luciano Burti had deputised for Eddie Irvine at the 2000 Austrian Grand Prix.
Sentence 1: Edmund Irvine Jr. (born 10 November 1965) is a retired British racing driver from Northern Ireland.
  Fact 1.0: Edmund Irvine Jr. was born on 10 November 1965.
  Fact 1.1: Edmund Irvine Jr. is a retired British racing driver.
  Fact 1.2: Edmund Irvine Jr. is from Northern Ireland.

Retrieved items:
<item 1>
Jordan 194
The Jordan 194 was a Formula One car for the 1994 season.

 The number 14 seat was taken by Rubens Barrichello and the number 15 seat was taken by Eddie Irvine.

 Kelvin Burt was named as test driver, but his mileage in the car was limited.

 The engine was a Hart 1035 3.5 V10, a developed version of the engine which had proved promising in 1993.

 The team's main sponsor was Sasol, with additional financial support coming from Arisco, the Irish government and other smaller sponsors.

 With driver aids such as traction control and active suspension banned for the 1994 season, the car used a conventional semi-automatic gearbox and pushrod
double wishbone suspension.

 The simpler technical regulations of 1994 seemed to benefit Jordan and the 194 recaptured the level of performance seen in the team's debut season, 1991.

 Indeed, the cut-down airbox and drooping front wing of the 194 recalled the design features of the 1991 car.
</item 1>

<item 2>
Spyker F1
The Spyker F1 Team, known as the Etihad Aldar Spyker F1 Team for sponsorship reasons was a Formula One team that competed in the 2007 Formula One World
Championship, and was created by Spyker Cars after their buyout of the short-lived Midland F1 (formerly Jordan Grand Prix) team.

 The change to the Spyker name was accompanied by a switch in racing livery from the red and white previously used by Midland, to an orange and silver
scheme—already seen on the Spyker Spyder GT2-R—orange being the national colour and the auto racing colour of the Netherlands.

 At the end of the 2007 season the team was sold and renamed Force India.
</item 2>

<item 3>
Jordan 195
The Jordan 195 was a Formula One car for the 1995 season.

 The number 14 seat was taken by Rubens Barrichello and the number 15 seat was taken by Eddie Irvine.

 The team never employed a test driver.

 The engine was a Peugeot A10 3.0 V10.

 The team's main sponsor was Total.
</item 3>

<item 4>
1999 British Grand Prix
The 1999 British Grand Prix was a Formula One motor race held on 11 July 1999 at the Silverstone Circuit near Silverstone, England.

 It was the eighth race of the 1999 Formula One season.

 The 60-lap race was won by McLaren driver David Coulthard after he started from third position.

 Eddie Irvine finished second for the Ferrari team and Williams driver Ralf Schumacher came in third.
</item 4>

<item 5>
Jaguar R2
The Jaguar R2 was the car with which the Jaguar team competed in the Formula One season.

 It was driven by Eddie Irvine, who was in his second year with the team, and Luciano Burti, who had deputised for Irvine at the 2000 Austrian Grand Prix.

 However, the Brazilian was replaced by Pedro de la Rosa after only four Grands Prix.
</item 5>

<item 6>
Graph query row:
person: Eddie Irvine
nationality: British
relationship: None
connected_entity: None
</item 6>

<item 7>
1983 Volvo International – Doubles
Sherwood Stewart and Ferdi Taygan were the defending champions but they competed with different partners that year, Stewart with Mark Edmondson and Taygan with
Cássio Motta.
</item 7>

<item 8>
1997 Argentine Grand Prix
The 1997 Argentine Grand Prix was a Formula One motor race held at Autódromo Oscar Alfredo Gálvez in Buenos Aires, Argentina on 13 April 1997.

 It was the third race of the 1997 Formula One season and the 600th World Championship event.

 The 72-lap race was won by Williams driver Jacques Villeneuve after he started from pole position.

 Eddie Irvine finished second for the Ferrari team and Jordan driver Ralf Schumacher came in third.
</item 8>

<item 9>
Eddie Irvine
Edmund Irvine Jr. (born 10 November 1965) is a retired British racing driver from Northern Ireland.

 He was a Formula One driver between 1993 and 2002, and runner-up in the 1999 World Drivers' Championship, driving for Scuderia Ferrari.
</item 9>

<item 10>
Jaguar R3
The Jaguar R3 is a Formula One racing car with which Jaguar Racing competed in the 2002 Formula One season.

 The car was launched on January 4, 2002.

 It was driven by Eddie Irvine and Pedro de la Rosa, both retained from 2001.
</item 10>
=== END ITEM id=edb0095ea1825935 ===

=== ITEM id=8c0c87d6eb166f98 ===
Gold sentences and facts:
Sentence 0: Tímea Babos (] ; born 10 May 1993) is a Hungarian tennis player.
  Fact 0.0: Tímea Babos was born on 10 May 1993.
  Fact 0.1: Tímea Babos is a Hungarian tennis player.
Sentence 1: Jack Sock (born September 24, 1992) is an American professional tennis player who is currently ranked world No. 21 in men's singles by the Association of Tennis Professionals (ATP).
  Fact 1.0: Jack Sock was born on September 24, 1992.
  Fact 1.1: Jack Sock is an American professional tennis player.
  Fact 1.2: Jack Sock is currently ranked world No. 21 in men's singles.
  Fact 1.3: The ranking is by the Association of Tennis Professionals (ATP).

Retrieved items:
<item 1>
Jack Sock
Jack Sock (born September 24, 1992) is an American professional tennis player who is currently ranked world No. 21 in men's singles by the Association of Tennis
Professionals (ATP).

 He is one of the top ranked American men in both singles and doubles on the ATP Tour.

 A former junior US Open champion, Sock's singles success is highlighted by 7 ATP finals, including three titles.

 His career best result came at the 2017 Indian Wells Masters, where he reached the semi-finals.
</item 1>

<item 2>
2012 Monterrey Open
The 2012 Monterrey Open was a women's tennis tournament played on outdoor hard courts.

 It was the 4th edition of the Monterrey Open and was an International tournament on the 2012 WTA Tour.

 It took place at the Sierra Madre Tennis Club in Monterrey, Mexico, from 20 to 26 February.

 Tímea Babos won the singles title.
</item 2>

<item 3>
Tímea Babos
Tímea Babos (] ; born 10 May 1993) is a Hungarian tennis player.
</item 3>

<item 4>
2013 Aegon Classic – Doubles
Tímea Babos and Hsieh Su-wei were the defending champions, but Babos decided not to participate.

<br>
</item 4>

<item 5>
2015 Apia International Sydney – Women's Doubles
Tímea Babos and Lucie Šafářová were the defending champions, but chose not to participate together.

 Babos played alongside Kristina Mladenovic, but lost in the semifinals to Raquel Kops-Jones and Abigail Spears.

 Šafářová teamed up with Belinda Bencic, but lost in the first round to Kimiko Date-Krumm and Karolína Plíšková.

<br>
</item 5>

<item 6>
2015 Malaysian Open – Doubles
Tímea Babos and Chan Hao-ching were the defending champions, but Chan chose not to participate this year.

 Babos chose to play in Monterrey, but lost in the first round.
</item 6>

<item 7>
2014 Suzhou Ladies Open – Doubles
Tímea Babos and Michaëlla Krajicek were the defending champions, having won the event in 2013, however both players chose not to participate.
</item 7>

<item 8>
Carly Gullickson
Carly Gullickson (born November 26, 1986) is a former American professional tennis player.
</item 8>

<item 9>
2016 Dubai Tennis Championships – Women's Doubles
Tímea Babos and Kristina Mladenovic were the defending champions, but chose not to compete together.

 Babos played alongside Julia Görges, but lost in the semifinals to Caroline Garcia and Mladenovic.

<br>
</item 9>

<item 10>
2014 Monterrey Open – Doubles
Tímea Babos and Kimiko Date-Krumm were the defending champions, but chose not to participate together.

 Babos played alongside Olga Govortsova, while Date-Krumm teamed up with Karolína Plíšková.

 The two teams were scheduled to meet in the semifinals, but Date-Krumm withdrew with a right leg injury.

<br>
</item 10>
=== END ITEM id=8c0c87d6eb166f98 ===
