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

This file has 11 items.

=== ITEM id=641d0ea742ac3647 ===
Gold sentences and facts:
Sentence 0: The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.
  Fact 0.0: The Battle of Saipan was a battle of the Pacific campaign of World War II.
  Fact 0.1: The Battle of Saipan was fought on the island of Saipan in the Mariana Islands.
  Fact 0.2: The Battle of Saipan was fought from 15 June to 9 July 1944.
Sentence 1:  The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
  Fact 1.0: The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division were commanded by Lieutenant General Holland Smith.
  Fact 1.1: The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division defeated the 43rd Infantry Division of the Imperial Japanese Army.
  Fact 1.2: The 43rd Infantry Division of the Imperial Japanese Army was commanded by Lieutenant General Yoshitsugu Saito.
Sentence 2:  It was fought by American and Filipino forces against Japanese troops in Manila, the capital city of the Philippines.
  Fact 2.0: The battle referred to here was fought by American and Filipino forces against Japanese troops.
  Fact 2.1: The battle referred to here took place in Manila, the capital city of the Philippines.

Retrieved items:
<item 1>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 1>

<item 2>
Battle of Bataan
The surrender at Bataan, with 76,000 soldiers surrendering in the Philippines altogether, was the largest in American and Filipino military histories, and was
the largest United States surrender since the American Civil War's Battle of Harper's Ferry.

 Soon afterwards, Filipino and American prisoners of war were forced into the Bataan Death March.
</item 2>

<item 3>
Battle of Manila (1945)
The Battle of Manila (February 3, 1945 – March 3, 1945) was a major battle of the Philippine campaign of 1944-45, during the Second World War.

 It was fought by American and Filipino forces against Japanese troops in Manila, the capital city of the Philippines.

 The month-long battle, which resulted in the death of over 100,000 civilians and the complete devastation of the city, was the scene of the worst urban
fighting in the Pacific theater.

 Japanese forces committed mass murder against Filipino civilians during the battle.

 Along with massive loss of life, the battle also destroyed architectural and cultural heritage dating back to the city's foundation.

 The battle ended the almost three years of Japanese military occupation in the Philippines (1942–1945).

 The city's capture was marked as General Douglas MacArthur's key to victory in the campaign of reconquest.
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
Battle of Okinawa
The Battle of Okinawa (Japanese: 沖縄戦 , Hepburn: Okinawa-sen ) (Okinawan: 沖縄戦 , "Uchinaa ikusa " ), codenamed Operation Iceberg, was a major battle of the
Pacific War fought on the island of Okinawa by United States Marine and Army forces against the Imperial Japanese Army and included the largest amphibious
assault in the Pacific War during World War II, the April 1, 1945, invasion of Okinawa itself.

 The 82-day battle lasted from April 1 until June 22, 1945.

 After a long campaign of island hopping, the Allies were planning to use Okinawa, a large island only 340 mi away from mainland Japan, as a base for air
operations for the planned invasion of the Japanese home islands.
</item 6>

<item 7>
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
</item 7>

<item 8>
Battle of Attu
The Battle of Attu, which took place from 11–30 May 1943, was a battle fought between forces of the United States, aided by Canadian reconnaissance and
fighter-bomber support, and the Empire of Japan on Attu Island off the coast of the Territory of Alaska as part of the Aleutian Islands Campaign during the
American Theater and the Pacific Theater and was the only land battle of World War II fought on incorporated territory of the United States.

 It is also the only land battle in which Japanese and American forces fought in Arctic conditions.
</item 8>

<item 9>
Battle of Guam (1944)
The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands
captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
</item 9>

<item 10>
Battle of Manila (1898)
The Battle of Manila, sometimes called the Mock Battle of Manila, was a land engagement which took place in Manila on August 13, 1898, at the end of the
Spanish–American War, four months after the decisive victory by Commodore Dewey's Asiatic Squadron at the Battle of Manila Bay.

 The belligerents were Spanish forces led by Governor-General of the Philippines Fermín Jáudenes, and American forces led by United States Army Brigadier
General Wesley Merritt and United States Navy Commodore George Dewey.

 American forces were supported by units of the Philippine Revolutionary Army, led by Emilio Aguinaldo.
</item 10>
=== END ITEM id=641d0ea742ac3647 ===

=== ITEM id=bd7aad6f2c7ff013 ===
Gold sentences and facts:
Sentence 0:  He earned 1 cap under Jack Charlton for the Irish national team.
  Fact 0.0: The player earned 1 cap for the Irish national team.
  Fact 0.1: The player earned that cap under Jack Charlton.
Sentence 1:  He was part of the England team that won the 1966 World Cup.
  Fact 1.0: The player was part of the England team that won the 1966 World Cup.
  Fact 1.1: The England team won the 1966 World Cup.

Retrieved items:
<item 1>
Billy Morgan (footballer, born 1891)
William Albert L. Morgan (3 November 1891 – after 1927) was an English professional footballer born in Old Hill, Cradley Heath, Staffordshire, who played either
at outside left or inside left.

 He played for Birmingham, Coventry City and Crystal Palace in the Football League, and was capped once for a Football League representative side.
</item 1>

<item 2>
William McCourty
William McCourty (1884 – 10 December 1917) was an English professional footballer born in Morpeth, Northumberland, who played in the Football League for
Birmingham.

 McCourty, a former coal miner who played at left half, joined Birmingham in May 1909.

 He made his debut in the Second Division on 13 September 1909, deputising for the injured Tommy Daykin in a home game against Glossop which finished as a 2–2
draw, and was never picked again, returning to non-league football the following year.
</item 2>

<item 3>
1929–30 British Home Championship
The 1929-30 British Home Championship was an edition of the annual international football tournament played between the British Home Nations.

 1930 was the year in which the tournament finally gained a serious rival as the premier international football competition, with the inception of the 1930 FIFA
World Cup, held in Uruguay.

 The Home Nations were not however members of FIFA due to disputes over the growing professionalism in continental and South American football.

 As a result, they were not able to attend and indicated that even if they were invited they would have no interest in attending, deeming foreign opposition too
weak for serious contest.

 It is interesting to speculate what would have happened had the Home Nations entered the tournament, especially as the tide of world football was changing
against Britain.

 The England team, which dominated the 1930 championship, had lost to Spain the year before in the first defeat by a foreign football team, and in the same year
they only managed draws with Germany and Austria.

 The Scottish side, which had won most of the previous ten championships, was likewise unprepared, only playing its first game outside the British Isles in
1929, and being heavily defeated on tour in 1931 by both the Austrians and the Italians.
</item 3>

<item 4>
Barry Murphy (footballer, born 1959)
Barry Murphy (born 1 April 1959 in Dublin) was an Irish soccer player during the 1980s.

 He earned 1 cap under Jack Charlton for the Irish national team.
</item 4>

<item 5>
Barry Rose
Barry Michael Rose OBE, FRAM, FRSCM, Hon.D.Mus, Hon.FGCM, Hon FRCO, M.Univ., (Surrey), born 24 May 1934) is a choir trainer and organist.

 He is best known for founding the choir and the pattern of daily sung Worship at the new Guildford Cathedral in 1961, and later, on 29 July 1981, directing the
music at the wedding of Charles, Prince of Wales and Diana, Princess of Wales formerly Lady Diana Frances Spencer, at St Paul's Cathedral in London.
</item 5>

<item 6>
Baron Plunket
Baron Plunket, of Newtown in the County of Cork, is a title in the Peerage of the United Kingdom.

 It was created in 1827 for the prominent Irish lawyer and Whig politician William Plunket.

 He served as Lord Chancellor of Ireland from 1830 and 1834 and again from 1835 to 1841.

 His eldest son, the second Baron, was Bishop of Tuam, Killala and Achonry between 1839 and 1866.

 He was succeeded by his younger brother, the third Baron.

 He was a barrister.

 His eldest son, the fourth Baron, served as Archbishop of Dublin between 1884 and 1897.

 He was succeeded by his eldest son, the fifth Baron.

 He was a diplomat and held office as Governor of New Zealand between 1904 and 1910.

 His grandson, Patrick, the seventh Baron, was Equerry to both King George VI and Queen Elizabeth II.

 He was succeeded by his younger brother, Robin who died in 2013 and was in turn succeeded by his nephew, Tyrone who was a Page of Honour to Queen Elizabeth II.
</item 6>

<item 7>
Dave Bailey (athlete)
He was a member of 9 Canadian National Track and Field Teams competing at World University Games, Budapest 1965 and Tokyo 1967, Commonwealth Games, Kingston,
Jamaica, 1966, Pan American Games, Winnipeg, 1967, Commonwealth vs USA, Los Angeles, 1967, Pre-Olympic Games, Mexico City, 1967, Soviet Union Tour, 1968,
European Tour, 1968, Olympic Games, Mexico City, 1968.
</item 7>

<item 8>
Clare Taylor
Clare Elizabeth Taylor MBE (born 22 May 1965) is an English sportswoman, the first woman to have played on a World Cup team in both cricket and football.

 She represented England at both cricket, as a member of the winning World Cup cricket team in 1993, and football (World Cup 1995).

 She was awarded the MBE in the Queen's Birthday Honours in 2000 for her services to women’s sport.

 Taylor attended Moor End High School and still has her name on the athletics records board.
</item 8>

<item 9>
Jack Charlton
John Charlton, OBE, DL (born 8 May 1935) is an English former footballer and manager who played as a defender.

 He was part of the England team that won the 1966 World Cup.

 He is the elder brother of former Manchester United forward Bobby Charlton, who was also a teammate in England's World Cup final victory.

 He spent his entire club career with Leeds United from 1950 to 1973, helping the club to the Second Division title (1963–64), First Division title (1968–69),
FA Cup (1972), League Cup (1968), Charity Shield (1969), Inter-Cities Fairs Cup (1968 and 1971), as well as one other promotion from the Second Division
(1955–56) and five second-place finishes in the First Division, two FA Cup final defeats and one Inter-Cities Fairs Cup final defeat.

 His 629 league and 762 total competitive appearances are club records.

 In 2006, Leeds United supporters voted Charlton into the club's greatest ever XI.
</item 9>

<item 10>
Walter Ward (footballer)
Walter Ward (1869 – after 1890) was an English professional footballer born in Birmingham who played in the Football Alliance for Small Heath.

 During the 1890–91 season, Ward deputised for regular goalkeeper Chris Charsley when Charsley's duties as a serving police officer demanded his absence from
the side.
</item 10>
=== END ITEM id=bd7aad6f2c7ff013 ===

=== ITEM id=0b4b63c087e8799c ===
Gold sentences and facts:
Sentence 0: The Silent Historian (original title: "Het zwijgen van Loe de Jong [The Silence of Loe de Jong]") is a 2011 Dutch documentary film by Simonka de Jong, shown as part of the January 2012 New York Jewish Film Festival, presented in partnership with Jewish Museum and Film Society of Lincoln Center.
  Fact 0.0: The Silent Historian has the original title 'Het zwijgen van Loe de Jong [The Silence of Loe de Jong]'.
  Fact 0.1: The Silent Historian is a 2011 Dutch documentary film.
  Fact 0.2: The Silent Historian is a documentary film by Simonka de Jong.
  Fact 0.3: The Silent Historian was shown as part of the January 2012 New York Jewish Film Festival.
  Fact 0.4: The January 2012 New York Jewish Film Festival screening of The Silent Historian was presented in partnership with the Jewish Museum.
  Fact 0.5: The January 2012 New York Jewish Film Festival screening of The Silent Historian was presented in partnership with the Film Society of Lincoln Center.
Sentence 1: The Betrayal — Nerakhoon is a 2008 documentary film directed by Ellen Kuras and Thavisouk Phrasavath.
  Fact 1.0: The Betrayal — Nerakhoon is a 2008 documentary film.
  Fact 1.1: The Betrayal — Nerakhoon was directed by Ellen Kuras.
  Fact 1.2: The Betrayal — Nerakhoon was directed by Thavisouk Phrasavath.

Retrieved items:
<item 1>
The Decembrists (film)
The Decembrists (Russian: Декабристы , "Dekabristy " ) is a 1927 Soviet silent historical drama film directed by Aleksandr Ivanovsky.
</item 1>

<item 2>
The entries highlight acclaimed Israeli cinema: *The Band’s Visit* (2007), directed by Eran Kolirin, and *Hunting Elephants* (2013), Reshef Levi’s crime comedy
starring Sasson Gabai, Moni Moshonov and Patrick Stewart. *Yana’s Friends* (1999) won 10 Ophir Awards and Karlovy Vary’s Crystal Globe. Joseph Cedar won Cannes’
2011 screenplay prize for *Footnote
</item 2>

<item 3>
The Counterfeiters (2007 film)
The Counterfeiters (German: "Die Fälscher" ) is a 2007 Austrian-German drama film written and directed by Stefan Ruzowitzky.

 It fictionalizes Operation Bernhard, a secret plan by Nazi Germany during World War II to destabilize the United Kingdom by flooding its economy with forged
Bank of England pound notes.

 The film centres on a Jewish counterfeiter, Salomon 'Sally' Sorowitsch, who is coerced into assisting the operation at the Sachsenhausen concentration camp.
</item 3>

<item 4>
The Price of Betrayal
The Price of Betrayal (Swedish: "Judaspengar" ) is a 1915 Swedish silent drama film directed by Victor Sjöström.
</item 4>

<item 5>
Austrian director-screenwriter Stefan Ruzowitzky made *The Inheritors* (1998), Austria’s unsuccessful Oscar submission, and *The Counterfeiters* (2007), a WWII
drama about Nazi forgery at Sachsenhausen that won the foreign-language-film Oscar; Devid Striesow played Bernhard Krüger. German director Wolfgang Becker is
known for *Good Bye, Lenin!* (2003), starring Daniel Brühl and Katrin Saß. Other entries
</item 5>

<item 6>
No Names on the Doors
No Names on the Doors (Hebrew: אין שמות על הדלתות‎ ) is a 1997 Israeli drama directed by Nadav Levitan.

 The film was screened at the Berlin International Film Festival, the Montpellier Film Festival, the Shanghai International Film Festival, and the Melbourne
International Film Festival.

 It received two nominations for an Ophir Award in the categories of Best Actor and Best Screenplay.
</item 6>

<item 7>
The Silent Historian
The Silent Historian (original title: "Het zwijgen van Loe de Jong [The Silence of Loe de Jong]") is a 2011 Dutch documentary film by Simonka de Jong, shown as
part of the January 2012 New York Jewish Film Festival, presented in partnership with Jewish Museum and Film Society of Lincoln Center.
</item 7>

<item 8>
Betrayal (1929 film)
Betrayal was a 1929 drama film produced for Famous Players-Lasky and released by Paramount Pictures.

 The film is the last silent film directed by Lewis Milestone, the last silent performance by Gary Cooper, the last silent performance by Germany's Emil
Jannings, and the only onscreen pairing of Cooper and Jannings.

 It is considered a lost film.
</item 8>

<item 9>
The Deal (2015 film)
The Deal (; lit.

 "Murder Request") is a 2015 South Korean crime thriller film directed by Son Yong-ho.
</item 9>

<item 10>
The Betrayal – Nerakhoon
The Betrayal — Nerakhoon is a 2008 documentary film directed by Ellen Kuras and Thavisouk Phrasavath.
</item 10>
=== END ITEM id=0b4b63c087e8799c ===

=== ITEM id=f08b8cd6908b6bea ===
Gold sentences and facts:
Sentence 0: Pietro Alessandro Gaspare Scarlatti (2 May 1660 – 22 October 1725) was an Italian Baroque composer, especially famous for his operas and chamber cantatas.
  Fact 0.0: Pietro Alessandro Gaspare Scarlatti was born on 2 May 1660.
  Fact 0.1: Pietro Alessandro Gaspare Scarlatti died on 22 October 1725.
  Fact 0.2: Pietro Alessandro Gaspare Scarlatti was an Italian Baroque composer.
  Fact 0.3: Pietro Alessandro Gaspare Scarlatti was especially famous for his operas.
  Fact 0.4: Pietro Alessandro Gaspare Scarlatti was especially famous for his chamber cantatas.
Sentence 1: Pyotr Ilyich Tchaikovsky ( ; Russian: Пётр Ильи́ч Чайко́вский ; 25 April/7 May 1840 – 25 October/6 November 1893), often anglicized as Peter Ilich Tchaikovsky, was a Russian composer of the romantic period, some of whose works are among the most popular music in the classical repertoire.
  Fact 1.0: The Russian name of Pyotr Ilyich Tchaikovsky is Пётр Ильи́ч Чайко́вский.
  Fact 1.1: Pyotr Ilyich Tchaikovsky was born on 25 April/7 May 1840.
  Fact 1.2: Pyotr Ilyich Tchaikovsky died on 25 October/6 November 1893.
  Fact 1.3: Pyotr Ilyich Tchaikovsky is often anglicized as Peter Ilich Tchaikovsky.
  Fact 1.4: Pyotr Ilyich Tchaikovsky was a Russian composer of the romantic period.
  Fact 1.5: Some works by Pyotr Ilyich Tchaikovsky are among the most popular music in the classical repertoire.
Sentence 2:  He was the first Russian composer whose music made a lasting impression internationally, bolstered by his appearances as a guest conductor in Europe and the United States.
  Fact 2.0: Tchaikovsky was the first Russian composer whose music made a lasting impression internationally.
  Fact 2.1: The international impression of Tchaikovsky's music was bolstered by his appearances as a guest conductor.
  Fact 2.2: Tchaikovsky appeared as a guest conductor in Europe.
  Fact 2.3: Tchaikovsky appeared as a guest conductor in the United States.

Retrieved items:
<item 1>
Tchaikovsky (film)
Tchaikovsky (Russian: Чайковский ) is a 1970 Soviet biopic film directed by Igor Talankin.

 It featured Innokenty Smoktunovsky in the role of the famous Russian composer Pyotr Ilyich Tchaikovsky.

 It was nominated for the 1971 Academy Award for Best Foreign Language Film as well as the Academy Award for Original Song Score and Adaptation.
</item 1>

<item 2>
Vladimir Ashkenazy
Vladimir Davidovich Ashkenazy (Russian: Влади́мир Дави́дович Ашкена́зи , "Vladimir Davidovich Ashkenazi"; born 6 July 1937) is an internationally recognized
solo pianist, chamber music performer, and conductor.

 He is originally from Russia and has held Icelandic citizenship since 1972.

 He has lived in Switzerland since 1978.

 Ashkenazy has collaborated with well-known orchestras and soloists.

 In addition, he has recorded a large storehouse of classical and romantic works.

 His recordings have earned him five Grammy awards plus Iceland's Order of the Falcon.
</item 2>

<item 3>
Piano Concerto No. 2 (Tchaikovsky)
Pyotr Ilyich Tchaikovsky's Piano Concerto No. 2 in G major, Op.

 44, was written in 1879–1880.

 It was dedicated to Nikolai Rubinstein, who had insisted he be allowed to perform it at the premiere as a way of making up for his harsh criticism of
Tchaikovsky's First Piano Concerto.

 Rubinstein was destined never to play it, however, as he died in March 1881.

 The premiere performance took place in New York City, on 12 November 1881.

 The soloist was Madeline Schiller, and Theodore Thomas conducted the New York Philharmonic orchestra.

 The first Russian performance was in Moscow in May 1882, conducted by Anton Rubinstein with Tchaikovsky's pupil, Sergei Taneyev, at the piano.
</item 3>

<item 4>
Chaikovskij (crater)
Chaikovskij (sometimes Tchaikovsky) is a crater on Mercury.

 It has a diameter of 165 kilometers.

 Its name was adopted by the International Astronomical Union in 1976.

 Chaikovskij is named for the Russian composer Pyotr Ilyich Tchaikovsky, who lived from 1840 to 1893.
</item 4>

<item 5>
Pyotr Ilyich Tchaikovsky
Pyotr Ilyich Tchaikovsky ( ; Russian: Пётр Ильи́ч Чайко́вский ; 25 April/7 May 1840 – 25 October/6 November 1893), often anglicized as Peter Ilich Tchaikovsky,
was a Russian composer of the romantic period, some of whose works are among the most popular music in the classical repertoire.

 He was the first Russian composer whose music made a lasting impression internationally, bolstered by his appearances as a guest conductor in Europe and the
United States.

 Tchaikovsky was honored in 1884, by Emperor Alexander III, and awarded a lifetime pension.
</item 5>

<item 6>
Kamarinskaya
Kamarinskaya (Russian: камаринская ) is a Russian traditional folk dance, which is mostly known today as the Russian composer Mikhail Glinka's composition of
the same name.

 Glinka's "Kamarinskaya", written in 1848, was the first orchestral work based entirely on Russian folk song and to use the compositional principles of that
genre to dictate the form of the music.

 It became a touchstone for the following generation of Russian composers ranging from the Western-oriented Pyotr Ilyich Tchaikovsky to the group of
nationalists known collectively as The Five and was also lauded abroad, most notably by French composer Hector Berlioz.
</item 6>

<item 7>
International Tchaikovsky Competition
The International Tchaikovsky Competition is a classical-music competition held every four years in Moscow, Russia, for pianists, violinists, and cellists
between 16 and 32 years of age, and singers between 19 and 32 years of age.

 The competition is named after Russian composer Pyotr Ilyich Tchaikovsky and is an active member of the World Federation of International Music Competitions.
</item 7>

<item 8>
Alessandro Scarlatti
Pietro Alessandro Gaspare Scarlatti (2 May 1660 – 22 October 1725) was an Italian Baroque composer, especially famous for his operas and chamber cantatas.

 He is considered the founder of the Neapolitan school of opera.

 He was the father of two other composers, Domenico Scarlatti and Pietro Filippo Scarlatti.
</item 8>

<item 9>
Aleksandr Ivanovsky
Aleksandr Ivanovsky (1881–1968) was a Russian screenwriter and film director who worked during the Soviet era.

 Ivanovsky was awarded the Stalin Prize in 1941, for his work on the 1940 film "Musical Story".

 His 1944 operetta film "Silva" was one of the most popular releases in the Soviet Union that year.
</item 9>

<item 10>
Piano Sonata in C-sharp minor (Tchaikovsky)
The Piano Sonata in C-sharp minor, Op.

 posth.

 80, was written by Russian composer Pyotr Ilyich Tchaikovsky in 1865, his last year as a student at the St Petersburg Conservatory.

 The sonata in its original form was not published in Tchaikovsky's lifetime; it was published in 1900 by P. Jurgenson, and given the posthumous opus number 80.
</item 10>
=== END ITEM id=f08b8cd6908b6bea ===

=== ITEM id=3e584da896a68c57 ===
Gold sentences and facts:
Sentence 0:  All matches were played at HC Davos's home arena, Eisstadion Davos.
  Fact 0.0: All matches were played at the home arena of HC Davos.
  Fact 0.1: The home arena of HC Davos is Eisstadion Davos.
Sentence 1: Hockey Club Davos is a professional Swiss ice hockey club based in Davos, Switzerland.
  Fact 1.0: Hockey Club Davos is a professional Swiss ice hockey club.
  Fact 1.1: Hockey Club Davos is based in Davos, Switzerland.

Retrieved items:
<item 1>
HC Davos
Hockey Club Davos is a professional Swiss ice hockey club based in Davos, Switzerland.

 Davos plays in the National League (NL).

 The team is usually a strong force in the league and often bolster their roster with Swiss national team players and players who once played for National
Hockey League teams.

 HC Davos also has hosted the annual Spengler Cup tournament, an invitational competition, since 1923.
</item 1>

<item 2>
2016 Spengler Cup
The 90th Spengler Cup was an ice hockey competition being held in Davos, Switzerland between 26 and 31 December 2016.

 All matches were played at HC Davos's home known as Vaillant Arena.

 Six competing teams were split into two groups of three (in the round-robin series).

 The two groups, named Torriani and Cattini, are named after legendary Swiss hockey players Richard 'Bibi' Torriani and the Cattini brothers, Hans and
Ferdinand.
</item 2>

<item 3>
2015 Spengler Cup
The 89th Spengler Cup was an ice hockey competition held in Davos, Switzerland between 26 and 31 December 2015.

 All matches were played at HC Davos's home known as Vaillant Arena.

 Six competing teams were split into two groups of three (in the round-robin series).

 The two groups, named Torriani and Cattini, are named after legendary Swiss hockey players Richard 'Bibi' Torriani and the Cattini brothers, Hans and
Ferdinand.
</item 3>

<item 4>
Spengler Cup
The Spengler Cup is an annual invitational ice hockey tournament held in Davos, Switzerland.

 First held in 1923, the Spengler Cup is often cited as the oldest invitational ice hockey tournament in the world.

 The event is hosted by the Swiss team HC Davos and played each year in Davos, Switzerland, between Christmas (December 25) and New Year's Day.

 Currently, all games are held at Vaillant Arena.
</item 4>

<item 5>
1992 Stanley Cup Finals
The 1992 Stanley Cup Finals NHL championship series was contested by the Prince of Wales Conference and defending Stanley Cup champion Pittsburgh Penguins and
the Clarence Campbell Conference champion Chicago Blackhawks.

 The Blackhawks were appearing in their first Finals since .

 After the Blackhawks jumped to an early 4–1 lead in the first game of the series, Mario Lemieux and the Penguins came back to win the game, sweep the series in
four games, and win their second consecutive and second overall Stanley Cup.

 It was the 99th year of the Stanley Cup, and the first to extend into the month of June.

 It was the last final for Chicago Stadium as it closed in 1994.
</item 5>

<item 6>
1992 Spengler Cup
The 1992 Spengler Cup was held in Davos, Switzerland, between December 26, 1992, and December 31, 1992.

 All matches were played at HC Davos's home arena, Eisstadion Davos.

 The final was won 6-5 by Team Canada over Färjestad BK.
</item 6>

<item 7>
Vaillant Arena
Vaillant Arena (Eisstadion Davos) is an arena in Davos, Switzerland.

 It is primarily used for ice hockey and is the home arena of HC Davos.

 It holds 7,080 people, of which 3,280 are seated.

 Every year the Spengler Cup is played in this arena.
</item 7>

<item 8>
Stadio del Ghiaccio Gianmario Scola
Stadio del Ghiaccio Gianmario Scola is an indoor sporting arena in Canazei, Italy.

 The capacity of the arena is 3,500 spectators.

 It is the home arena of the Sportiva Hockey Club Fassa ice hockey team of the Serie A ice hockey league.

 The arena also hosted games during the 1994 World Ice Hockey Championships along with the Forum di Assago in Milan and the PalaOnda in Bolzano.
</item 8>

<item 9>
Montreal Victorias
The Victoria Hockey Club of Montreal, Quebec, Canada was an early men's amateur ice hockey club.

 Its date of origin is ascribed to either 1874, 1877 or 1881, making it either the first or second organized ice hockey club after McGill University.

 The club played at its own rink, the Victoria Skating Rink in Montreal.

 The club was winners of the Stanley Cup in 1895 and held it except for a period in 1896 until 1899.

 The club remained amateur, splitting from the ranks of teams turned professional in 1908.

 The amateur hockey club was the first winner of the Allan Cup and continued in play until 1939 after its 65th season.

 The club often also fielded junior and intermediate teams.
</item 9>

<item 10>
1984 Stanley Cup Finals
This would be the second of eight consecutive Finals contested by a team from Alberta (the Oilers appeared in six, the Calgary Flames in two), and the first of
five consecutive Finals to end with the Cup presentation on Alberta ice (the Oilers won four times, the Montreal Canadiens one).
</item 10>
=== END ITEM id=3e584da896a68c57 ===

=== ITEM id=13456666c33e0353 ===
Gold sentences and facts:
Sentence 0: Sir Ridley Scott (born 30 November 1937) is an English film director and producer.
  Fact 0.0: Sir Ridley Scott was born on 30 November 1937.
  Fact 0.1: Sir Ridley Scott is English.
  Fact 0.2: Sir Ridley Scott is a film director.
  Fact 0.3: Sir Ridley Scott is a film producer.
Sentence 1:  Following his commercial breakthrough with the science-fiction horror film "Alien" (1979), his best known works include the neo-noir dystopian science fiction film "Blade Runner" (1982), historical drama and Best Picture Oscar winner "Gladiator" (2000), and science fiction film "The Martian" (2015).
  Fact 1.0: Ridley Scott had a commercial breakthrough with the science-fiction horror film Alien, released in 1979.
  Fact 1.1: Ridley Scott's best known works include the neo-noir dystopian science fiction film Blade Runner, released in 1982.
  Fact 1.2: Ridley Scott's best known works include the historical drama Gladiator, released in 2000.
  Fact 1.3: Gladiator won the Best Picture Oscar.
  Fact 1.4: Ridley Scott's best known works include the science fiction film The Martian, released in 2015.
Sentence 2:  A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work behind the camera, with Griffith and Joseph Henabery as his mentors.
  Fact 2.0: Ridley Scott was a collaborator of D.W. Griffith.
  Fact 2.1: Ridley Scott appeared in The Birth of a Nation, released in 1915.
  Fact 2.2: Ridley Scott appeared in Intolerance, released in 1916.
  Fact 2.3: Ridley Scott gave up acting in 1917.
  Fact 2.4: Ridley Scott gave up acting to concentrate on work behind the camera.
  Fact 2.5: Griffith was a mentor of Ridley Scott.
  Fact 2.6: Joseph Henabery was a mentor of Ridley Scott.

Retrieved items:
<item 1>
Elmer Clifton
Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.

 A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work
behind the camera, with Griffith and Joseph Henabery as his mentors.

 His first feature-length solo effort as a director was "The Flame of Youth" with Jack Mulhall.
</item 1>

<item 2>
The entries profile filmmakers, producers, performers, and artists: Tim Bedore is a Chicago-born comedian; Rogue One (2016) depicts rebels stealing Death Star
plans; Frederick K. Keller, Peter Guber, Baltasar Kormákur, Steve Rash, Jonathan Kaplan, Kurt Voss, Jim Sharman, Barry Sandler, and Howard Rosenman are
directors or producers. Other highlights include Titan A.E.’s commercial failure; Vito Russo’s documentary; Scott Free’s founding by the
</item 2>

<item 3>
Ridley Scott
Sir Ridley Scott (born 30 November 1937) is an English film director and producer.

 Following his commercial breakthrough with the science-fiction horror film "Alien" (1979), his best known works include the neo-noir dystopian science fiction
film "Blade Runner" (1982), historical drama and Best Picture Oscar winner "Gladiator" (2000), and science fiction film "The Martian" (2015).
</item 3>

<item 4>
Scott Free Productions
Scott Free Productions is a British film and television production company founded by filmmakers and brothers Ridley Scott and Tony Scott.

 They formed the feature film development company Percy Main Productions in 1980, naming the company after the English village Percy Main, where their father
grew up.

 The company was renamed to Scott Free Productions in 1995.

 Scott Free has produced films ranging from the 2000 Hollywood blockbuster "Gladiator" (2000) to "smaller pictures" like "Cracks" (2009).

 Between productions of "White Squall" (1996) and "G.I. Jane" (1997), Ridley Scott reorganized the company.
</item 4>

<item 5>
Steve McQueen (director)
Steven Rodney "Steve" McQueen (born 9 October 1969) is an English film director, producer, screenwriter, and video artist.

 For his 2013 film, "12 Years a Slave", a historical drama adaptation of an 1853 slave narrative memoir, he won an Academy Award, BAFTA Award for Best Film, and
Golden Globe Award for Best Motion Picture – Drama, as a producer, and he also received the award for Best Director from the New York Film Critics Circle.

 McQueen is the first black filmmaker to win an Academy Award for Best Picture.

 McQueen is known for his collaborations with actor Michael Fassbender, who has starred in all three of McQueen's feature films as of 2014.

 McQueen's other feature films are "Hunger" (2008), a historical drama about the 1981 Irish hunger strike, and "Shame" (2011), a drama about an executive
struggling with sex addiction.
</item 5>

<item 6>
Dan Ouellette
Giger and Beksinski, and he has been featured in magazines internationally.

 David Bowie commented while looking at Dan's art that he has noticed a strong influence of science fiction on contemporary art.

 Giger saw a different aspect, saying simply "Very bony."

 He grew to adore the cinema of Fellini and later to marvel at Lynch's "Eraserhead".

 Rather than pursue a formal education in the fine arts he chose to study the craft of film making and after college he became a production designer working on
feature films in New York City starting with his work as a production designer for Hal Hartley in 1990 with Trust and then, in 1992, with Simple Men.
</item 6>

<item 7>
Rick Carter
Rick Carter (born 1950) is an American production designer and art director.

 He is known for his work in the film "Forrest Gump", which earned him an Oscar nomination, as well as numerous nominations of other awards for his work in
"Amistad" and "A.I. Artificial Intelligence".

 Other films include "Cast Away", "War of the Worlds", "What Lies Beneath", "Jurassic Park", "Avatar", and "Back to the Future Part II" and "Part III".

 Many of the films that he has worked on are directed by Steven Spielberg or Robert Zemeckis.

 For his part in the Art Direction of "Avatar", he was awarded the Academy Award for Best Production Design alongside Robert Stromberg and Kim Sinclair.

 In 2013, Carter won his second Academy Award, for production design on Steven Spielberg's biopic, "Lincoln".
</item 7>

<item 8>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 8>

<item 9>
Shirley Clarke
Shirley Clarke (October 2, 1919 – September 23, 1997) was an American experimental and independent filmmaker.

 She was also a director and editor to many famous works.
</item 9>

<item 10>
Spencer Averick
Spencer Averick is an American film editor and producer.

 Best known for his work an editor on critically acclaimed films "Middle of Nowhere" (2012), "Selma" (2014) and for producing 2016 acclaimed documentary "13th"
for which he received Academy Award for Best Documentary Feature nominations at 89th Academy Awards, that he shared with director Ava DuVernay and co-producer
Howard Barish.
</item 10>
=== END ITEM id=13456666c33e0353 ===

=== ITEM id=345b33bdff6cfcf5 ===
Gold sentences and facts:
Sentence 0: Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.
  Fact 0.0: Semyon Aronovich Gershgorin was born on August 24, 1901.
  Fact 0.1: Semyon Aronovich Gershgorin died on May 30, 1933.
  Fact 0.2: Semyon Aronovich Gershgorin was Soviet.
  Fact 0.3: Semyon Aronovich Gershgorin was born in Pruzhany, Belarus, Russian Empire.
  Fact 0.4: Semyon Aronovich Gershgorin was a mathematician.
Sentence 1:  He was a mathematician noted for his work on the decidability of various algebraic groups. Malcev algebras (generalisations of Lie algebras) are named after him.
  Fact 1.0: Semyon Aronovich Gershgorin was a mathematician.
  Fact 1.1: Semyon Aronovich Gershgorin was noted for his work on the decidability of various algebraic groups.
  Fact 1.2: Malcev algebras are generalisations of Lie algebras.
  Fact 1.3: Malcev algebras are named after Semyon Aronovich Gershgorin.

Retrieved items:
<item 1>
AES+F
AES+F is an artist collective of four Russian artists: Tatiana Arzamasova (1955), Lev Evzovich (1958), Evgeny Svyatsky (1957), and Vladimir Fridkes (1956).

 It was first formed as AES Group in 1987 by Arzamasova, Evzovich, and Svyatsky, becoming AES+F when Fridkes joined in 1995.

 The collective works in photography, video, installation, and animation, as well as more traditional media, such as painting, drawing, and sculpture.

 AES+F's early work included performance, installation, painting, and illustration.

 Well known for their monumental video-art installations that Gareth Harris describes as "monumental painting set in motion", AES+F create grand visual
narratives that explore contemporary global values, vices and conflicts.
</item 1>

<item 2>
Semyon Aranovich Gershgorin
Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.

 He began as a student at the Petrograd Technological Institute in 1923, became a Professor in 1930, and was given an appointment at the Leningrad Mechanical
Engineering Institute in the same year.

 His contributions include the Gershgorin circle theorem.
</item 2>

<item 3>
Anna Aglatova
Anna Khachaturovna Aglatova (Russian: Анна Хачатуровна Аглатова ) is a Russian soprano singer who was born in Kislovodsk and by 2004 joined the singing
department of the Gnessin State Musical College.

 Prior to it, she was a recipient of the "Sergei Leiferkus grant" from the Vladimir Spivakov Fund and in 2005 made her first public appearance at the Bolshoi
Theatre.

 Her career there, have not started with ease since her father didn't want her to work for Bolshoi.

 In 2003, she was a recipient of the first prize at the Bella voce International Competition and the same year participated at both the Christmas Festival at
Düsseldorf and the fourteenth annual Chaliapin Season at Mineralnye Vody.

 She also was a participant for "Irina Arkhipova Fund" and was a part of the Novosibirsk Opera and Ballet Theatre.
</item 3>

<item 4>
Nikolay Mitrofanovich Krylov
Nikolay Mitrofanovich Krylov (Russian: Никола́й Митрофа́нович Крыло́в , Ukrainian: Микола Митрофанович Крилов ) (29 November [O.S. 17 November] 1879 – May 11,
1955) was a Russian and Soviet mathematician known for works on interpolation, non-linear mechanics, and numerical methods for solving equations of mathematical
physics.
</item 4>

<item 5>
Vladimir Arnold
Vladimir Igorevich Arnold (alternative spelling Arnol'd, Russian: Влади́мир И́горевич Арно́льд , 12 June 1937 – 3 June 2010) was a Soviet and Russian
mathematician.

 While he is best known for the Kolmogorov–Arnold–Moser theorem regarding the stability of integrable systems, he made important contributions in several areas
including dynamical systems theory, catastrophe theory, topology, algebraic geometry, symplectic geometry, differential equations, classical mechanics,
hydrodynamics and singularity theory, including posing the ADE classification problem, since his first main result—the solution of Hilbert's thirteenth problem
in 1957 at the age of 19.
</item 5>

<item 6>
Anatoly Maltsev
Anatoly Ivanovich Maltsev (also: Malcev, Mal'cev; Russian: Анато́лий Ива́нович Ма́льцев; 27 November N.S./14 November O.S. 1909, Moscow Governorate – 7 June
1967, Novosibirsk) was born in Misheronsky, near Moscow, and died in Novosibirsk, USSR.

 He was a mathematician noted for his work on the decidability of various algebraic groups. Malcev algebras (generalisations of Lie algebras) are named after
him.
</item 6>

<item 7>
Aleksandr Ivanovsky
Aleksandr Ivanovsky (1881–1968) was a Russian screenwriter and film director who worked during the Soviet era.

 Ivanovsky was awarded the Stalin Prize in 1941, for his work on the 1940 film "Musical Story".

 His 1944 operetta film "Silva" was one of the most popular releases in the Soviet Union that year.
</item 7>

<item 8>
Nicolai V. Krylov
Nicolai Vladimirovich Krylov (Russian: Никола́й Влади́мирович Крыло́в ; born 5 June 1941) is a Russian mathematician specializing in partial differential
equations, particularly stochastic partial differential equations and diffusion processes.

 Krylov studied at Lomonosov University, where he in 1966 under E. B. Dynkin attained a doctoral candidate title (similar to a PhD) and in 1973 a Russian
doctoral degree (somewhat more prestigious than a PhD).

 He taught from 1966 to 1990 at the Lomonosov University and is since 1990 a professor at the University of Minnesota.

 At the beginning of his career (starting from 1963) he, in collaboration with Dynkin, worked on nonlinear stochastic control theory, making advances in the
study of convex, nonlinear partial equations of 2nd order ("i.e." Bellman equations), which were examined with stochastic methods.
</item 8>

<item 9>
Anatoly Fomenko
Anatoly Timofeevich Fomenko (Russian: Анато́лий Тимофе́евич Фоме́нко ) (born 13 March 1945 in Stalino, USSR) is a Soviet and Russian mathematician, professor at
Moscow State University, well known as a topologist, and a member of the Russian Academy of Sciences.

 He is author of a pseudoscientific theory known as New Chronology.

 He is also a member of the Russian Academy of Natural Sciences (1991).
</item 9>

<item 10>
Alexander Artemev
Alexander Vladimirovich "Sasha" Artemev (Russian: Александр Владимирович "Саша" Артемьев , Belarusian: Аляксандр Уладзіміравіч Арцем'еў ; born August 29, 1985)
is a retired American artistic gymnast.

 Artemev was a member of the bronze medal-winning U.S. team at the 2008 Olympic Games.

 He is the 2006 all-around U.S. national champion.

 Known for his ability on pommel horse, he is the 2007 and 2008 U.S. national champion on the pommel horse and won the bronze medal on the event at the 2006
World Championships.
</item 10>
=== END ITEM id=345b33bdff6cfcf5 ===

=== ITEM id=c067deb1b3101f86 ===
Gold sentences and facts:
Sentence 0: Swingin' Down Yonder is the first full-length, 12-inch album recorded by Dean Martin for Capitol Records during three sessions in September and October 1954 and February 1955.
  Fact 0.0: Swingin' Down Yonder is the first full-length, 12-inch album recorded by Dean Martin for Capitol Records.
  Fact 0.1: Dean Martin recorded Swingin' Down Yonder during three sessions.
  Fact 0.2: The sessions for Swingin' Down Yonder took place in September 1954, October 1954 and February 1955.
Sentence 1:  One of the most popular and enduring American entertainers of the mid-20th century, Martin was nicknamed the "King of Cool" for his seemingly effortless charisma and self-assurance.
  Fact 1.0: Dean Martin was one of the most popular and enduring American entertainers of the mid-20th century.
  Fact 1.1: Dean Martin was nicknamed the 'King of Cool'.
  Fact 1.2: Dean Martin's nickname 'King of Cool' reflected his seemingly effortless charisma.
  Fact 1.3: Dean Martin's nickname 'King of Cool' reflected his seemingly effortless self-assurance.

Retrieved items:
<item 1>
Swingin' Down Yonder
Swingin' Down Yonder is the first full-length, 12-inch album recorded by Dean Martin for Capitol Records during three sessions in September and October 1954 and
February 1955.

 According to the original sleeve notes, all the songs have a "common geographical root: the American South."

 In 1963, Capitol Records re-released "Swingin' Down Yonder" under the titled "Southern Style".
</item 1>

<item 2>
Here Comes Louis Smith
Here Comes Louis Smith is the debut album by American trumpeter Louis Smith recorded in 1958 and released on the Blue Note label.

 Originally recorded for the Transition label, the company went out of business shortly afterwards and before the recording could be released.

 The album masters were acquired by Blue Note's Alfred Lion.
</item 2>

<item 3>
Things Ain't What They Used to Be
In 1941 there was a strike against the American Society of Composers, Authors and Publishers, of which Duke Ellington was a member.

 Because of the strike he could not air his songs on the radio.

 Instead, he used songs written by his son Mercer and pianist Billy Strayhorn.

 Strayhorn's compositions of this time include "Take the 'A' Train", "Chelsea Bridge" and "Day Dream", and Mercer wrote "Things Ain't What They Used to Be",
"Blue Serge" and "Moon Mist".
</item 3>

<item 4>
Willie and the Hand Jive
"Willie and the Hand Jive" is a song written by Johnny Otis and originally released as a single in 1958 by Johnny Otis Show, reaching #9 on the "Billboard" Hot
100 chart and #5 on the Billboard R&B chart.

 The song has a Bo Diddley beat and was partly inspired by the music sung by a chain gang Otis heard while he was touring.

 The lyrics are about a man who became famous for doing a dance with his hands, but the song has been accused of glorifying masturbation.

 It has since been covered by numerous artists, including The Strangeloves, Eric Clapton, Cliff Richard, Kim Carnes, George Thorogood and The Grateful Dead.

 Clapton's 1974 version was also released as a single and also reached the Billboard Top 40, peaking at #26.

 Thorogood's 1985 version reached #25 on the Hot Mainstream Rock Tracks chart.
</item 4>

<item 5>
Basie Jazz
Basie Jazz is an album by pianist/bandleader Count Basie recorded in 1952 and released on the Clef label in 1954.

 Selections from this album were also released on the 1956 Clef LPs "The Swinging Count!

" and "Basie Rides Again!

".
</item 5>

<item 6>
Cecil Gant
Cecil Gant (April 4, 1913 – February 4, 1951) was an American blues singer, songwriter and pianist, whose recordings of both ballads and "fiery piano rockers"
were successful in the mid- and late 1940s, and influenced the early development of rock and roll.

 His biggest hit was the 1944 ballad, "I Wonder".
</item 6>

<item 7>
Graph query row:
entity: Dean Martin
aliases: ['Dino Paul Crocetti', 'King of Cool']
entity_types: ['__KGBuilder__', 'Person', '__Entity__']
kind: None
published_on: None
occupation: singer
nationality: Italian-American
relationship: MEMBER_OF
related: Rat Pack
</item 7>

<item 8>
Jack Johnson (album)
Jack Johnson, later reissued as A Tribute to Jack Johnson, is a 1971 studio album and soundtrack by American jazz trumpeter, composer, and bandleader Miles
Davis.

 In 1970, Davis was asked by Bill Cayton to record music for his documentary of the same name on the life of boxer Jack Johnson.

 Johnson's saga resonated personally with Davis, who wrote in the album's liner notes of Johnson's mastery as a boxer, his affinity for fast cars, jazz,
clothes, and beautiful women, his unreconstructed blackness, and his threatening image to white men.

 This was the second film score he had composed, after "Ascenseur pour l'échafaud" in 1957.
</item 8>

<item 9>
One O'Clock Jump
"One O'Clock Jump" is a jazz standard, a 12-bar blues instrumental, written by Count Basie in 1937.

 The melody derived from band members' riffs--Basie rarely wrote down musical ideas, so Eddie Durham and Buster Smith helped him crystallize his ideas.

 The original 1937 recording of the tune by Basie and his band is noted for the saxophone work of Herschel Evans and Lester Young, trumpet by Buck Clayton,
Walter Page on bass and Basie himself on piano.
</item 9>

<item 10>
Collins &amp; Harlan
Collins & Harlan, the team of Arthur Collins and Byron G. Harlan formed a popular comic duet between 1903 and 1926.

 They sang ragtime standards as well as what were known as "Coon songs" – music sung by white performers in a black dialect.

 Their material also employed many other stereotypes of the time including Irishmen and farmers.

 Fellow recording artist Billy Murray nicknamed them "The Half-Ton Duo" as both men were rather overweight.

 Collins and Harlan produced many number one hits with recordings of minstrel songs such as "My Gal Irene", "I Know Dat I'll be Happy Til I Die", "Who Do You
Love?"

 and "Down Among the Sugarcane".

 Their song "That Funny Jas Band from Dixieland", recorded November 8, 1916, is among the first recorded uses of the word "jas" which eventually evolved to
"jass", and to the current spelling "jazz".
</item 10>
=== END ITEM id=c067deb1b3101f86 ===

=== ITEM id=1d1f96536d5c058c ===
Gold sentences and facts:
Sentence 0: Cherie Ann Currie (born November 30, 1959) is an American musician (instrumentalist, singer, songwriter), actress and artist.
  Fact 0.0: Cherie Ann Currie was born November 30, 1959.
  Fact 0.1: Cherie Ann Currie is an American musician.
  Fact 0.2: Cherie Ann Currie is an instrumentalist, singer, and songwriter.
  Fact 0.3: Cherie Ann Currie is an actress.
  Fact 0.4: Cherie Ann Currie is an artist.
Sentence 1: Steve Augeri (born January 30, 1959) is an American rock singer best known for his work with Tall Stories, Tyketto, and Journey.
  Fact 1.0: Steve Augeri was born January 30, 1959.
  Fact 1.1: Steve Augeri is an American rock singer.
  Fact 1.2: Steve Augeri is best known for his work with Tall Stories.
  Fact 1.3: Steve Augeri is best known for his work with Tyketto.
  Fact 1.4: Steve Augeri is best known for his work with Journey.

Retrieved items:
<item 1>
Reverie (Cherie Currie album)
Reverie is the third full-length studio album by Cherie Currie.

 Released on iTunes March 16, 2015.

 Cherie released the CD version of this album June 5, 2015 on her eBay page cheriecurriedirect.

 There is a 35-year gap between Cherie's last full-length studio album, 1980's Messin' with the Boys (with Marie Currie), and 2015's Reverie.

 This is last studio album Kim Fowley produced before his death.

 Kim helped Cherie release this album to make amends with her after all the money he swindled her out of when she was in the Runaways and for releasing her and
Marie's music on Young and Wild without their approval.

 After Kim's death Cherie's son, Jake Hays, took over producing.
</item 1>

<item 2>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 2>

<item 3>
Young and Wild (album)
Young and Wild is a compilation by Cherie & Marie Currie.

 This album has all 10 original tracks from Messin' with the Boys, six songs from Beauty's Only Skin Deep, three songs Cherie Currie sang with The Runaways, and
one new track co-written by Marie Currie, "Longer Than Forever".

 "Longer Than Forever" was the B side of the single "Since You Been Gone".
</item 3>

<item 4>
Age of Chance
Age of Chance were a British alternative rock-dance crossover band from Leeds, England active from 1985 to 1991.

 They were perhaps most known for their mutant metallic cover of Prince's "Kiss" which topped the UK Indie Chart in 1986, and peaked at No. 50 in the UK Singles
Chart in January the following year.

 Despite signing for major label Virgin, and being favourites with the UK music press, they never enjoyed a major hit in the UK, although "Don't Get Mad… Get
Even" reached No. 8 on the US "Billboard" Hot Dance/Club Play chart.
</item 4>

<item 5>
Marie Currie
Marie Michelle Currie (born November 30, 1959) is an American singer, songwriter, actress, and artist.

 Currie is best known for playing in a band with her identical twin Cherie Currie called Cherie & Marie Currie.

 Their song "Since You Been Gone" charted at number 95 on the US charts.

 Marie played Singing Maid Marie in "The Rosebud Beach Hotel" and is now a popular multi-media sculptress and artist.
</item 5>

<item 6>
Richie Furay
Paul Richard "Richie" Furay (born May 9, 1944) is an American singer, songwriter, and Rock & Roll Hall of Fame member who is best known for forming the bands
Buffalo Springfield with Stephen Stills, Neil Young, Bruce Palmer, and Dewey Martin, and Poco with Jim Messina, Rusty Young, George Grantham and Randy Meisner.

 His best known song (originally written during his tenure in Buffalo Springfield, but eventually performed by Poco, as well) was "Kind Woman", which he wrote
for his wife, Nancy.
</item 6>

<item 7>
Cherie Currie
Cherie Ann Currie (born November 30, 1959) is an American musician (instrumentalist, singer, songwriter), actress and artist.

 Currie was the lead vocalist of the Runaways, a rock band from Los Angeles, in the mid-to-late 1970s.

 After the Runaways, she became a solo artist.

 Then she teamed up with her identical twin sister, Marie Currie, and released an album with her.

 They released a duet "Since You Been Gone" which charted number 95 on US charts.

 Their band was called Cherie and Marie Currie.

 She is also well known for her role in the movie "Foxes".
</item 7>

<item 8>
Messin' with the Boys
Messin' with the Boys is the second post-Runaways album by Cherie Currie, released in 1980 For this album Currie worked with her identical twin sister, Marie
Currie.

 Their band was called "Cherie and Marie Currie".

 Marie was a guest vocalist on Cherie's first album, so Marie went on tour with Cherie to support her first album, "Beauty's Only Skin Deep".

 When Marie would join Cherie on stage to sing the encores the audience would go wild.

 So Cherie ran with the idea of two blonds are better than one.

 The idea paid off because "Messin' with the Boys" received more radio play than "Beauty's Only Skin Deep", and their song "Since You Been Gone" made it to 95
on US charts.

 The single "This Time" and the album Messin' with the Boys made the top 200 on U.S. charts.

 This makes Messin' with the Boys Cherie Currie's most successful album.

 "I Just Love the Feeling" originally surfaced on the 1974 album, "S.S.
</item 8>

<item 9>
Steve Augeri
Steve Augeri (born January 30, 1959) is an American rock singer best known for his work with Tall Stories, Tyketto, and Journey.
</item 9>

<item 10>
Cherushii
Cherushii (Chelsea Faith Dolan) (September 14, 1983 – December 2, 2016) was an American electronic music producer, DJ, live performer, and radio host based in
San Francisco.

 Influenced by house music, Detroit techno, and disco, she released several recordings on the independent dance music label 100% Silk, including "Queen of
Cups", "Far Away So Close", and "Memory of Water", reissued in 2017.

 KQED named Cherushii's album "Meow Wolf’s Arcade Soundtracks: Wiggy’s Plasma Plex" one of the 10 best bay area albums of 2016.

 While Cherushii has received little recognition from mainstream media, her music is of enormous influence in the underground music scene.

 Fellow electronic musician Maria Minerva has called Cherushii criminally underrated.
</item 10>
=== END ITEM id=1d1f96536d5c058c ===

=== ITEM id=f2b12d08c07495ef ===
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
The Ipcress File (film)
The Ipcress File is a 1965 British espionage film directed by Sidney J. Furie, starring Michael Caine and featuring Guy Doleman and Nigel Green.

 The screenplay by Bill Canaway and James Doran was based on Len Deighton's novel, "The IPCRESS File" (1962).

 It has won critical acclaim and a BAFTA award for best British film.

 In 1999 it was included at number 59 on the BFI list of the 100 best British films of the 20th century.
</item 1>

<item 2>
Sheila Levine Is Dead and Living in New York
Sheila Levine Is Dead and Living in New York is a 1975 film directed by Sidney J. Furie, about a shy young woman who moves to New York City and falls in love
with the boyfriend of her extroverted roommate.

 The film was co-written by Kenny Solms and Gail Parent, and based on her novel.

 The film was shot on location in New York City.
</item 2>

<item 3>
The Rocky Horror Picture Show
The Rocky Horror Picture Show is a 1975 British-American musical screwball comedy horror film by 20th Century Fox produced by Lou Adler and Michael White and
directed by Jim Sharman.

 The screenplay was written by Sharman and actor Richard O'Brien, who appears in the film, which is based on the 1973 musical stage production "of the same
title", with music, book, and lyrics by O'Brien.

 The production is a parody tribute to the science fiction and horror B movies of the 1930s through to the early 1970s.

 Along with O'Brien, the film stars Tim Curry, Susan Sarandon, and Barry Bostwick and is narrated by Charles Gray with cast members from the original Royal
Court Theatre, Roxy Theatre, and Belasco Theatre productions.
</item 3>

<item 4>
Shadows of the Night (film)
Shadows of the Night is a 1928 American drama silent film directed by D. Ross Lederman and written by Robert E. Hopkins and D. Ross Lederman.

 The film stars Flash the Dog, Lawrence Gray, Louise Lorraine, Warner Richmond and Tom Dugan.

 It was released on October 26, 1928, by Metro-Goldwyn-Mayer.
</item 4>

<item 5>
The Snake Woman
The Snake Woman is a low budget 1961 British horror film directed by Sidney J. Furie and starring Susan Travers, John McCarthy.
</item 5>

<item 6>
The Boys in Company C
The Boys in Company C, directed by Sidney J. Furie, starring Stan Shaw, Andrew Stevens, Craig Wasson, James Canning, and Michael Lembeck, is a 1978 film about
United States Marine Corps recruits preparing for duty, and their subsequent combat in the Vietnam War.

 It was among the first Vietnam War films to appear after the Vietnam Era, and was also the first role for R. Lee Ermey of "Full Metal Jacket" fame.

 "The Boys in Company C" is the first in Furie's Vietnam War motion picture trilogy, followed by 2001's "Under Heavy Fire" and 2006's "The Veteran".
</item 6>

<item 7>
Doris Bither case
The Doris Bither case, also known as the Entity hauntings, was an alleged haunting which occurred in 1974 at Culver City, California where a woman named Doris
Bither alleged the ghosts of three Asian men were raping her.

 The case inspired Frank De Felitta's 1978 book "The Entity" which was made into a 1982 film of the same name starring Barbara Hershey.
</item 7>

<item 8>
The Entity
The Entity is a 1982 American horror film directed by Sidney J. Furie and written by Frank De Felitta, who adapted his 1978 novel of the same name.

 It stars Barbara Hershey as a woman who is raped and tormented by an invisible assailant.

 Despite being filmed and planned for a release in 1981, the movie was not released in worldwide theaters until September 1982 and February 1983 in the United
States.

 Like the novel, the film is based on the 1974 Doris Bither case.
</item 8>

<item 9>
Doctor Blood's Coffin
Doctor Blood's Coffin is a 1961 British horror film directed by Sidney J. Furie.

 Kieron Moore stars as a mad scientist who reanimates the dead.
</item 9>

<item 10>
Little Fauss and Big Halsy
Little Fauss and Big Halsy is a 1970 film directed by Sidney J. Furie, starring Robert Redford and Michael J. Pollard, also featuring Lauren Hutton, Noah Beery,
Jr. and Lucille Benson.
</item 10>
=== END ITEM id=f2b12d08c07495ef ===

=== ITEM id=71eaf3d95ac441a3 ===
Gold sentences and facts:
Sentence 0: In September, The Fader published an interview with CupcakKe, which she discussed her new album that would address "rape, and everything else".
  Fact 0.0: The Fader published an interview with CupcakKe in September.
  Fact 0.1: In the interview, CupcakKe discussed her new album.
  Fact 0.2: CupcakKe's new album would address "rape, and everything else".
Sentence 1: The Fader (stylized as The FADER) is a New York City-based music magazine launched in 1999 by Rob Stone and Jon Cohen, covering music, style and culture.
  Fact 1.0: The Fader, stylized as The FADER, is a New York City-based music magazine.
  Fact 1.1: The Fader was launched in 1999.
  Fact 1.2: The Fader was launched by Rob Stone and Jon Cohen.
  Fact 1.3: The Fader covers music, style and culture.
Sentence 2:  It was the first print publication to be released on iTunes.
  Fact 2.0: The Fader was the first print publication to be released on iTunes.

Retrieved items:
<item 1>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 1>

<item 2>
The Fader
The Fader (stylized as The FADER) is a New York City-based music magazine launched in 1999 by Rob Stone and Jon Cohen, covering music, style and culture.

 It was the first print publication to be released on iTunes.
</item 2>

<item 3>
State (magazine)
State.ie (formerly State Magazine) is an Irish website and formerly a monthly music magazine, which launched in April 2008 and ceased to print in January 2009
having published a total of ten issues.

 The magazine continues online and was voted Best Music Site in 2008 and Best Web Publication in 2010 in the Irish Web Awards.

 Originally the concept of the magazine involved a hard copy of which there was a charge to purchase, however after six issues it was decided to make the
magazine's content free both online and in print.

 The first issue, April 2008, appeared on Irish shelves on 6 March 2008 and featured Michael Stipe of R.E.M. on the cover.

 This immediately garnered comparisons between the new magazine's similarities with "Hot Press" who featured Stipe on their cover at the same time, a move
widely thought to be an attempt by Hot Press to stifle State's status as a serious 'alternative' to the more established local magazine.

 At a price of €5.50, "State" charged €2 more than "Hot Press".
</item 3>

<item 4>
Exclaim!
Exclaim!

 is a monthly Canadian music magazine that features in-depth coverage of new music across all genres with a special focus on Canadian and cutting-edge artists.

 Content is based on the monthly print publication, which publishes 9 issues per year, distributing over 103,000 copies to over 2,600 locations across Canada.

 The magazine has an average of 361,200 monthly readers.

 Their website, exclaim.ca, has an average of 675,000 unique visitors a month.
</item 4>

<item 5>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 5>

<item 6>
Paper (magazine)
Paper, stylized as PAPER, is a New York City-based independent magazine focusing on fashion, popular culture, nightlife, music, art and film.

 Past cover models include Kim Kardashian, Demi Lovato, Katy Perry, Miley Cyrus, Prince, CL, and Jennifer Lopez.
</item 6>

<item 7>
Rap-Up
Rap-Up is a magazine launched in 2001 by founder Devin Lazerine.

 The publication was originally a website devoted to hip hop, until Lazerine decided to pitch the possibility of a magazine to several publishers.

 The magazine is focused on the hip hop and R&B aspect of the music industry, and predominantly features interviews with artists, actors and other entertainers.

 Issued quarterly, the magazine's target audience was the 14 to 28 demographic, although the demographic has become older over time.
</item 7>

<item 8>
VIVmag
VIVmag is one of the earliest exclusively digital, interactive women's lifestyle magazines.

 Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio
Systems, Inc.

 The first issue appeared in January-February 2007.

 There is no print version.

 The paperless title can be viewed in most browsers or downloaded to your computer and read via the Zinio Reader.

 The magazine is based in Thousand Oaks, California.
</item 8>

<item 9>
Audacious (CupcakKe album)
In September, The Fader published an interview with CupcakKe, which she discussed her new album that would address "rape, and everything else".
</item 9>

<item 10>
SexIs
SexIs Magazine is a quarterly print publication and daily webzine devoted to sex and sexual culture, founded in 2008 by Web Merchants, parent company of sex toy
e-tailer EdenFantasys.com.

 The first print issue debuted in November 2009, distributed nationwide as an insert in "BUST magazine".

 The website publishes articles, columns, video presentations and news items daily.
</item 10>
=== END ITEM id=71eaf3d95ac441a3 ===
