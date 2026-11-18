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

=== ITEM id=a5598ef0d27885dd ===
Gold sentences and facts:
Sentence 0: Innocence is a 2013 American horror drama film directed by Hilary Brougher, who co-wrote the film with Tristine Skyler.
  Fact 0.0: Innocence is a 2013 American horror drama film.
  Fact 0.1: Innocence was directed by Hilary Brougher.
  Fact 0.2: Hilary Brougher co-wrote Innocence with Tristine Skyler.
Sentence 1:  The movie stars Sophie Curtis, Kelly Reilly, Graham Phillips, Linus Roache, Sarah Sutherland and Stephanie March.
  Fact 1.0: The movie Innocence stars Sophie Curtis.
  Fact 1.1: The movie Innocence stars Kelly Reilly.
  Fact 1.2: The movie Innocence stars Graham Phillips.
  Fact 1.3: The movie Innocence stars Linus Roache.
  Fact 1.4: The movie Innocence stars Sarah Sutherland.
  Fact 1.5: The movie Innocence stars Stephanie March.
Sentence 2: Stephanie Caroline March (born (1974--) 23, 1974 ) is an American actress.
  Fact 2.0: Stephanie Caroline March was born on 23, 1974.
  Fact 2.1: Stephanie Caroline March is an American actress.
Sentence 3:  She is best known for her role as Alexandra Cabot in the long-running NBC series, "".
  Fact 3.0: Stephanie March is best known for her role as Alexandra Cabot.
  Fact 3.1: The role of Alexandra Cabot was in a long-running NBC series.

Retrieved items:
<item 1>
Conviction (2006 TV series)
Conviction is an American television drama on NBC that debuted as a mid-season replacement on Friday, March 3, 2006.

 The cast includes Stephanie March reprising her "" role as Alexandra Cabot.

 In the series, Cabot returns to New York City and becomes a Bureau Chief ADA supervising a group of young but talented assistant district attorneys after a
stint in the Witness Protection Program.

 Other cast members include Eric Balfour, Anson Mount, Jordan Bridges, Julianne Nicholson, Milena Govich, and J. August Richards.
</item 1>

<item 2>
Kim Johnston Ulrich
Kim Johnston Ulrich (born March 24, 1955) is an American actress.

 From 1983 to 1986, Ulrich played the role of Diana McColl on "As the World Turns".

 In 1988, she guest-starred in two episodes of "Werewolf".

 In 1990, Ulrich appeared on the TV series "Wings" as Carol, Brian's ex-wife.

 In 1993, Ulrich appeared in the pilot episode of "" as Dr. Antoinette Baines, a villainous scientist.

 She appeared in a third season episode of "Highlander: The Series" in 1995.

 She appeared in the first episode of season 4 of "" as a conniving wife in "Murder by Friendly Fire" in 1996.
</item 2>

<item 3>
Stephanie March
Stephanie Caroline March (born (1974--) 23, 1974 ) is an American actress.

 She is best known for her role as Alexandra Cabot in the long-running NBC series, "".
</item 3>

<item 4>
Gloria Garayua
Gloria Garayua (born October 18, 1978 in New York City) is an American film and television actress.

 Garayua made her major film debut in the 2005 comedy film "Fun With Dick and Jane", and is now commonly cast in guest roles on long-running series such as "Six
Feet Under", "Weeds" and "The Shield".

 After being cast in an ongoing role in "Grey's Anatomy", Garayua has played recurring roles in other popular TV shows such as "Cougar Town" and "How to Get
Away with Murder".
</item 4>

<item 5>
Anastasia Horne
Anastasia Nicole "Ana" Horne (born August 5, 1978 in Claremont, California) is a former American actress and singer who has appeared on television and in the
theater.

 Her last television role was "Lori" on the MTV series "Undressed" in 2000.

 She also starred as "Lark Madison-Scanlon" on the daytime soap opera "Port Charles" from 1997 to 1999.

 As a teenager, she was cast as "Ana" on the long-running NBC/The Disney Channel series "Kids Incorporated" from 1984, to 1985, and 1991 to the end of the
series in 1994.
</item 5>

<item 6>
Innocence (2013 film)
Innocence is a 2013 American horror drama film directed by Hilary Brougher, who co-wrote the film with Tristine Skyler.

 The movie is based on the 2000 novel of the same name by Jane Mendelsohn.

 It had its world premiere on 26 October, 2013 at the Austin Film Festival and received a limited theatrical release in the United States on 5 September, 2014.

 The movie stars Sophie Curtis, Kelly Reilly, Graham Phillips, Linus Roache, Sarah Sutherland and Stephanie March.
</item 6>

<item 7>
Rachelle Lefevre
Rachelle Lefevre ( ; born February 1, 1979) is a Canadian actress.

 She has starred in the television series "Big Wolf on Campus" and had recurring roles in "What About Brian", "Boston Legal", and "Swingtown".

 She played the vampire Victoria Sutherland in the first two films of the "Twilight" saga, before being replaced by Bryce Dallas Howard in the third
installment, "", due to scheduling conflicts with her role in the film "Barney's Version".

 In 2011, she starred in the ABC medical drama "Off the Map", followed by the CBS series "A Gifted Man" (2011–2012) and "Under the Dome" (2013–2015).
</item 7>

<item 8>
Allison Miller
Allison Miller (born September 2, 1985) is an American actress.

 She is best known for playing Michelle Benjamin on the NBC series "Kings", Skye Tate on the Fox series "Terra Nova", and Carrie on the NBC series "Go On".

 She starred as Laura Larson on the Syfy television series "Incorporated".
</item 8>

<item 9>
Tempestt Bledsoe
Tempestt Bledsoe (born August 1, 1973) is an American actress.

 She is best known for her childhood role as Vanessa Huxtable, the fourth child of Cliff and Clair Huxtable on the long-running NBC sitcom "The Cosby Show"
(1984–92).

 In December 2010, it was announced that Bledsoe would be the host of "Clean House" on the Style Network, replacing long-time host Niecy Nash.

 From September 2012 to February 2013, she was one of the stars of the NBC TV sitcom "Guys with Kids", portraying Marny.
</item 9>

<item 10>
Sherry Alberoni
Sharyn Eileen "Sherry" Alberoni (born December 4, 1946) is an American actress and voice artist.

 Alberoni got her start as a Mouseketeer on the weekday ABC television program "The Mickey Mouse Club".

 As an adult, she became a voice artist for Hanna-Barbera Productions.

 Besides providing voices for numerous incidental characters in series such as "Jeannie", Alberoni is best known as the voice of nasty rich-girl Alexandra Cabot
from "Josie and the Pussycats" , "superhero-in-training" Wendy from the first season of "Super Friends", the heroic robot, Bo in "Mighty Orbots" and
Glumdalclitch in "The Three Worlds of Gulliver".

 In 1971, she starred alongside Patty Andrews in the Sherman Brothers stage musical, "Victory Canteen".
</item 10>
=== END ITEM id=a5598ef0d27885dd ===

=== ITEM id=1b95484b5f7b67ff ===
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
Battle of Attu
The Battle of Attu, which took place from 11–30 May 1943, was a battle fought between forces of the United States, aided by Canadian reconnaissance and
fighter-bomber support, and the Empire of Japan on Attu Island off the coast of the Territory of Alaska as part of the Aleutian Islands Campaign during the
American Theater and the Pacific Theater and was the only land battle of World War II fought on incorporated territory of the United States.

 It is also the only land battle in which Japanese and American forces fought in Arctic conditions.
</item 1>

<item 2>
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
</item 2>

<item 3>
Battle of Guam (1944)
The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands
captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
</item 3>

<item 4>
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
</item 4>

<item 5>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 5>

<item 6>
Operation Diadem order of battle
Operation Diadem order of battle is a listing of the significant formations that were involved in the fighting on the Winter Line and at the Anzio bridgehead
south of Rome during "Operation Diadem" in May - June 1944 which resulted in the Allied breakthrough at Cassino and the breakout at Anzio leading to the capture
of Rome.
</item 6>

<item 7>
Operation Epsom
Operation Epsom, also known as the First Battle of the Odon, was a British Second World War offensive that took place between 26 and 30 June 1944, during the
Battle of Normandy.

 The offensive was intended to outflank and seize the German-occupied city of Caen, an important Allied objective, in the early stages of Operation Overlord,
the Allied invasion of north-west Europe.
</item 7>

<item 8>
Operation Kikusui
Operation Kikusui (菊水作戦 , Kikusui sakusen ) was a series of suicidal air attacks by Imperial Japanese forces during the Battle of Okinawa against Allied fleets
in the waters around Okinawa, as part of Operation Ten-Go.

 The name of the operation, "Kikusui", comes from the hata-jirushi of the samurai Kusunoki Masashige.
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
Operation Diadem
Operation Diadem, also referred to as the Fourth Battle of Monte Cassino or, in Canada, the Battle of the Liri Valley, was an offensive operation undertaken by
the Allies of World War II (U.S. Fifth Army and British Eighth Army in May 1944, as part of the Italian Campaign of World War II.

 "Diadem" was supported by air attacks called Operation Strangle.

 The opposing force was the German 10th Army.
</item 10>
=== END ITEM id=1b95484b5f7b67ff ===

=== ITEM id=2c446691ad56766d ===
Gold sentences and facts:
Sentence 0: Victor Ronald Salva (born March 29, 1958) is an American film director and screenwriter.
  Fact 0.0: Victor Ronald Salva was born March 29, 1958.
  Fact 0.1: Victor Ronald Salva is an American film director.
  Fact 0.2: Victor Ronald Salva is an American screenwriter.
Sentence 1: Emilio "El Indio" Fernández (born Emilio Fernández Romo, ] ; March 26, 1904 – August 6, 1986) was a Mexican film director, actor and screenwriter.
  Fact 1.0: Emilio "El Indio" Fernández was born Emilio Fernández Romo.
  Fact 1.1: Emilio "El Indio" Fernández was born March 26, 1904.
  Fact 1.2: Emilio "El Indio" Fernández died August 6, 1986.
  Fact 1.3: Emilio "El Indio" Fernández was a Mexican film director.
  Fact 1.4: Emilio "El Indio" Fernández was a Mexican actor.
  Fact 1.5: Emilio "El Indio" Fernández was a Mexican screenwriter.

Retrieved items:
<item 1>
Antonio De Carlo
Antonio De Carlo, born (1967--) 4, 1967 (age 50 ) in Tijuana, Baja California, Mexico, is a Mexican actor also known as Antonio in the late 1980s and early
1990s.

 He won an Emmy Award in 2005-2006.

 He is president and founder of the "Fundación Cultura Sin Fronteras AC".

 After twelve years out of show business, in 2012 he returned to the soap operas performing "Magic Dragon" one of the main characters for "Miss XV", a
co-production from Televisa México and Nickelodeon directed and produced by Pedro Damian.

 He is also part of the cast of "Muchacha Italiana Viene a Casarse", directed by Pedro Damian (Televisa, México).
</item 1>

<item 2>
Jaime Fernández (actor)
Jaime Fernández Reyes (December 6, 1937 – April 15, 2005) was a Mexican actor.

 Over his career, he won 3 Silver Ariel awards - the Mexican equivalent of the Oscar - including one for what is arguably his best-known role, playing Friday in
Luis Buñuel's "Robinson Crusoe".

 He appeared in over 200 films and served as the General Secretary of the Mexican actors' union for 11 years.

 His older brother was actor/director Emilio Fernández.
</item 2>

<item 3>
Uruguayan cinema has grown since the late 1990s, producing over 120 fiction and nonfiction films with international recognition. Mexico’s Golden Age, especially
the 1940s–50s, featured director Emilio Fernández, whose *María Candelaria* won Cannes’ Palme d’Or; his films included *Enamorada*, *Las Abandonadas*, and
*Bugambilia*. Actors Jaime and Fernando Fernández were his brothers; Jaime won three
</item 3>

<item 4>
Victor Salva
Victor Ronald Salva (born March 29, 1958) is an American film director and screenwriter.

 He directed the films "Powder" (1995) and "Jeepers Creepers" (2001).

 The latter of these has been expanded into a franchise with two sequels that he has also directed.
</item 4>

<item 5>
Valerio Fioravanti
Giuseppe Valerio "Giusva" Fioravanti (born 28 March 1958) is an Italian who, with Francesca Mambro, was a leading figure in a far right terrorist group "Nuclei
Armati Rivoluzionari" ("Armed Revolutionary Nuclei" or NAR).

 Fioravanti appeared in films and television from a young age, and in his early teens was the most famous child in Italy.

 He and Mambro were fugitives wanted for terrorist offences by their early twenties, they spent a further period on the run while wanted for the Bologna
bombing.

 Both were captured after gunfights with police.

 Although Fioravanti, Mambro and a third NAR mamber were convicted of the bombing, there were those who thought that a higher level of the conspiracy was never
uncovered.

 Fioravanti and Mambro admit carrying out terrorist murders, but have always denied any involvement in the Bologna station bomb.
</item 5>

<item 6>
Filiberto Ojeda Ríos
Filiberto Ojeda Ríos (April 26, 1933 – September 23, 2005) was the commander-in-chief ("Responsable General") of the Boricua Popular Army ("Ejército Popular
Boricua, a.k.a., Los Macheteros").

 According to an unsigned article in the "Los Angeles Times", Los Macheteros was “a terrorist group seeking Puerto Rico's independence.”

 The group campaigned for, and supported, the independence of Puerto Rico from the United States.

 In 2001, FBI Director Louis J. Freeh claimed the group was linked to acts of terrorism, but Ronald Fernandez, scholar of Puerto Rican history, suggests such
labeling was an act of political convenience by the United States Government, intended to "shift the blame for any attacks on U.S. policy or personnel from us
to them".

 Ríos was also a founder of the FALN.

 In a 1983 New York Times article, Robert McFadden described the FALN as a Puerto Rican terrorist organization responsible for bombings during the 1970s and
early 1980s "in the name of Puerto Rican independence".
</item 6>

<item 7>
Emilio Fernández
Emilio "El Indio" Fernández (born Emilio Fernández Romo, ] ; March 26, 1904 – August 6, 1986) was a Mexican film director, actor and screenwriter.

 He was one of the most prolific film directors of the Golden Age of Mexican cinema in the 1940s and 1950s.

 He is best known for his work as director of the film "Maria Candelaria" (1944), which won the Palme d'Or award at the 1946 Cannes Film Festival.

 As an actor, he worked in numerous film productions in Mexico and in Hollywood.
</item 7>

<item 8>
Comedy in the Golden Age of Mexican cinema
Sadly, it was not as successful as many directors and film maker though it would be.

 Many failed to correctly synchronize the audio with the picture.

 Although to some extent it may have been unsuccessful, it is agreeable that it helped revolutionize film in Latin America.

 Mexico’s film industry takes its height post “El Cine Sonoro” to what is known as “The Golden Age”.

 Many film members consider this one of the most successful eras in the history of Mexican cinema.

 Many recognize the films of this era when they see black and white.

 It is also important to note that the films also portrayed social issues.

 Some of the most influential directors and actors are Emilio Fernández, Raul De Anda, Roberto B. Cervenna, Cantinflas, Vitola, Tin Tan, etc.

 The focus of this article will be on some of top grossing comedian actors of the time Cantinflas, Tin Tan, Clavillazo and others.
</item 8>

<item 9>
Fernando Fernández (actor)
Fernando Fernández "El Crooner de México" (1916–1999) was a Mexican actor and singer.

 He was born on November 9, 1916, in Monterrey, Mexico, and died in Mexico City, Mexico, on November 24, 1999.

 Fernando was the son of Eloisa Reyes.

 He was brother of the famous film director Emilio Fernández and actor Jaime Fernández.

 He was married to singer Lupita Palomera, who died in 2008.

 Fernando Fernández was known as "the Crooner of Mexico."
</item 9>

<item 10>
Columba Domínguez
Columba Domínguez Adalid (March 4, 1929 – August 13, 2014) was a Mexican actress, singer, and painter.

 Considered a crucial figure in the Golden Age of Mexican cinema, she was one of the muses of the film director Emilio Fernández, with whom she was romantically
linked for several years.

 She is remembered particularly for her performance in the film "Pueblerina" (1949), which is considered one of the jewels of the Mexican Cinema.
</item 10>
=== END ITEM id=2c446691ad56766d ===

=== ITEM id=200746afdf55cc2a ===
Gold sentences and facts:
Sentence 0: Jean-François André Clervoy (born 19 November 1958) is a French engineer and a CNES and ESA astronaut.
  Fact 0.0: Jean-François André Clervoy was born on 19 November 1958.
  Fact 0.1: Jean-François André Clervoy is a French engineer.
  Fact 0.2: Jean-François André Clervoy is a CNES astronaut.
  Fact 0.3: Jean-François André Clervoy is an ESA astronaut.
Sentence 1: Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
  Fact 1.0: Prof. Dr. Ulrich Hans Walter was born on February 9, 1954.
  Fact 1.1: Prof. Dr. Ulrich Hans Walter is a German physicist.
  Fact 1.2: Prof. Dr. Ulrich Hans Walter is a German engineer.
  Fact 1.3: Prof. Dr. Ulrich Hans Walter is a former DFVLR astronaut.

Retrieved items:
<item 1>
Ulrich Walter
Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
</item 1>

<item 2>
Jean-François Clervoy
Jean-François André Clervoy (born 19 November 1958) is a French engineer and a CNES and ESA astronaut.

 He is a veteran of three NASA Space Shuttle missions.
</item 2>

<item 3>
Jean-Loup Chrétien
Jean-Loup Jacques Marie Chrétien (born 20 August 1938) is a French retired "Général de Brigade" (brigadier general) in the "Armée de l'Air" (French air force),
and a former CNES spationaut.

 He flew on two Franco-Soviet space missions and a NASA Space Shuttle mission.

 Chrétien was the first Frenchman and the first western European in space.
</item 3>

<item 4>
Ernst Messerschmid
Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
</item 4>

<item 5>
Sigmund Jähn
Sigmund Werner Paul Jähn (born 13 February 1937) is a German cosmonaut and pilot, who in 1978 became the first East German (and German native) to fly in space
as part of the Soviet Union's Interkosmos programme.
</item 5>

<item 6>
Claude Nicollier
Claude Nicollier (born 2 September 1944 in Vevey, Switzerland) is the first astronaut from Switzerland.

 He has flown on four Space Shuttle missions.

 His first spaceflight (STS-46) was in 1992, and his final spaceflight (STS-103) was in 1999.

 He took part in two servicing missions to the Hubble Space Telescope (called STS-61 and STS-103).

 During his final spaceflight he participated in a spacewalk, becoming the first European Space Agency astronaut to do so during a Space Shuttle mission
(previous ESA astronauts conducted spacewalks aboard "Mir", see List of spacewalks and moonwalks 1965–1999).

 In 2000 he was assigned to the Astronaut Office Extravehicular Activity Branch, while maintaining a position as Lead ESA Astronaut in Houston.

 Nicollier retired from ESA in April 2007.
</item 6>

<item 7>
Ulf Merbold
Dr. Ulf Dietrich Merbold (born June 20, 1941) is the first West German citizen and second German native (after Sigmund Jähn) to have flown in space.

 He is also the first member of the European Space Agency Astronaut Corps to participate in a spaceflight mission and the first non-US citizen to reach orbit in
a US spacecraft.

 In 1983, he and Byron Lichtenberg became the first Payload Specialists to fly on the shuttle.
</item 7>

<item 8>
Patrick Baudry
Patrick Pierre Roger Baudry (born March 6, 1946 in Cameroon), is a retired Lieutenant Colonel in the French Air Force and a former CNES astronaut.

 In 1985, he became the second French citizen in space, after Jean-Loup Chrétien, when he flew aboard NASA's Space Shuttle mission STS-51-G.
</item 8>

<item 9>
Hans Schlegel
Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
</item 9>

<item 10>
Walter Cunningham
Ronnie Walter Cunningham (born March 16, 1932), (Col, USMCR, Ret.)

, better known as Walter Cunningham, is a retired American astronaut.

 In 1968, he was a Lunar Module Pilot on the Apollo 7 mission.

 He was NASA's third civilian astronaut (after Neil Armstrong and Elliot See), and has also been a fighter pilot, physicist, entrepreneur, venture capitalist,
author of "The All-American Boys", lecturer, and host of the radio show "Lift-off to Logic".

 Walter Cunningham attended UCLA.
</item 10>
=== END ITEM id=200746afdf55cc2a ===

=== ITEM id=89cd9cff6ca4d5e4 ===
Gold sentences and facts:
Sentence 0: The Karlsschrein (English: Shrine of Charlemagne ) in Aachen Cathedral was made in Aachen at the command of Frederick II, Holy Roman Emperor and completed in 1215, after Frederick II's grandfather, Frederick Barbarossa had exhumed Charlemagne's bones from their resting place in the Palatine Chapel, Aachen in 1165.
  Fact 0.0: The Karlsschrein, English name Shrine of Charlemagne, is in Aachen Cathedral.
  Fact 0.1: The Karlsschrein was made in Aachen.
  Fact 0.2: The Karlsschrein was made at the command of Frederick II, Holy Roman Emperor.
  Fact 0.3: The Karlsschrein was completed in 1215.
  Fact 0.4: Frederick II's grandfather was Frederick Barbarossa.
  Fact 0.5: Frederick Barbarossa exhumed Charlemagne's bones from their resting place in the Palatine Chapel, Aachen.
  Fact 0.6: Frederick Barbarossa exhumed Charlemagne's bones in 1165.
  Fact 0.7: The exhumation of Charlemagne's bones happened before the Karlsschrein was completed.
Sentence 1: Frederick I (German: "Friedrich I" , Italian: "Federico I" ; 1122 – 10 June 1190), also known as Frederick Barbarossa (Italian: "Federico Barbarossa" ), was the Holy Roman Emperor from 1155 until his death.
  Fact 1.0: Frederick I's German name is "Friedrich I".
  Fact 1.1: Frederick I's Italian name is "Federico I".
  Fact 1.2: Frederick I lived from 1122 to 10 June 1190.
  Fact 1.3: Frederick I is also known as Frederick Barbarossa.
  Fact 1.4: Frederick Barbarossa's Italian name is "Federico Barbarossa".
  Fact 1.5: Frederick I was the Holy Roman Emperor from 1155 until his death.

Retrieved items:
<item 1>
Harald II of Denmark
Harald II of Denmark (died 1018) was King of Denmark from 1014 until his death in 1018.

 He was the youngest son of Sweyn Forkbeard and Gunhild of Wenden, and was regent while his father was fighting Ethelred the Unready in England.

 He inherited the Danish throne in 1014, and held it while his brother, the later king Cnut the Great conquered England.

 After his death in 1018, he was succeeded by Cnut the Great.

 Little detail is known about Harald II.
</item 1>

<item 2>
Ernest Augustus, King of Hanover
Ernest Augustus (5 June 1771 – 18 November 1851) was King of Hanover from 20 June 1837 until his death.

 He was the fifth son and eighth child of King George III of the United Kingdom and Hanover.

 As a fifth son, Ernest seemed unlikely to become a monarch, but none of his four elder brothers had a legitimate son who survived infancy.

 The Salic Law, which barred succession to or through a female, prevailed in Hanover; therefore, when his elder brother King William IV died in 1837, Ernest
succeeded him as King of Hanover.

 In the United Kingdom the succession to the monarchy was determined by primogeniture and his niece Victoria succeeded to the throne, thus ending the personal
union between the British Isles and Hanover that had existed since 1714.
</item 2>

<item 3>
Eric III of Denmark
Eric III Lamb (Danish: "Erik III Lam" ) (  1120 – 27 August 1146) was the King of Denmark from 1137 until 1146.

 He was the grandson of Eric I of Denmark and the nephew of Eric II of Denmark, whom he succeeded on the throne.

 He abdicated in 1146, as the first and only Danish monarch to do so.

 His succession led to a period of civil war between Sweyn III, Canute V, and Valdemar I.
</item 3>

<item 4>
Battle of Parma
The Battle of Parma was fought on 18 February 1248 between the forces of Holy Roman Emperor Frederick II and the Guelphs.

 The Guelphs attacked the Imperial camp when Frederick II was away.

 The Imperial forces were defeated and much of Frederick's treasure was lost.
</item 4>

<item 5>
List of Holy Roman Empresses
Holy Roman Empress or Empress of the Holy Roman Empire is the title given to the consort (wife) or regent of the Holy Roman Emperor.

 The elective dignity of Holy Roman Emperor was restricted to males only, therefore there was never a Holy Roman Empress regnant, though women such as Theophanu
or Maria Theresa of Austria, who controlled the power of rule, served as de facto Empresses regnant.
</item 5>

<item 6>
Karl Anselm, 4th Prince of Thurn and Taxis
Karl Anselm, 4th Prince of Thurn and Taxis, full German name: "Karl Anselm Fürst von Thurn und Taxis" (2 June 1733, Frankfurt am Main, Free Imperial City of
Frankfurt, Holy Roman Empire – 13 November 1805, Winzer bei Regensburg, Electorate of Bavaria, Holy Roman Empire) was the fourth Prince of Thurn and Taxis,
Postmaster General of the Imperial Reichspost, and Head of the Princely House of Thurn and Taxis from 17 March 1773 until his death on 13 November 1805.

 Karl Anselm served as "Prinzipalkommissar" at the Perpetual Imperial Diet in Regensburg for Joseph II, Holy Roman Emperor and Francis II, Holy Roman Emperor
from 1773 to 1797.
</item 6>

<item 7>
Karlsschrein
The Karlsschrein (English: Shrine of Charlemagne ) in Aachen Cathedral was made in Aachen at the command of Frederick II, Holy Roman Emperor and completed in
1215, after Frederick II's grandfather, Frederick Barbarossa had exhumed Charlemagne's bones from their resting place in the Palatine Chapel, Aachen in 1165.
</item 7>

<item 8>
Constance, Queen of Sicily
Constance (2 November 1154 – 27 November 1198) was Queen regnant of Sicily in 1194–98, jointly with her spouse from 1194 to 1197, and with her infant son
Frederick II, Holy Roman Emperor, in 1198, as the heiress of the Norman kings of Sicily.

 She was also Holy Roman Empress by marriage to Henry VI, Holy Roman Emperor.
</item 8>

<item 9>
Frederick I, Holy Roman Emperor
Frederick I (German: "Friedrich I" , Italian: "Federico I" ; 1122 – 10 June 1190), also known as Frederick Barbarossa (Italian: "Federico Barbarossa" ), was the
Holy Roman Emperor from 1155 until his death.

 He was elected King of Germany at Frankfurt on 4 March 1152 and crowned in Aachen on 9 March 1152.

 He became King of Italy in 1155 and was crowned Roman Emperor by Pope Adrian IV on 18 June 1155.

 Two years later, the term "sacrum " ("holy") first appeared in a document in connection with his Empire.

 He was later formally crowned King of Burgundy, at Arles on 30 June 1178.

 He was named "Barbarossa " by the northern Italian cities which he attempted to rule: Barbarossa means "red beard" in Italian; in German, he was known as
"Kaiser Rotbart ", which has the same meaning.
</item 9>

<item 10>
Eric II of Denmark
Eric II the Memorable (Danish: "Erik II Emune" ) (  1090 – 18 September 1137) was king of Denmark between 1134 and 1137.

 Eric was an illegitimate son of Eric I of Denmark, who ruled Denmark from 1095 to 1103.

 Eric the Memorable rebelled against his uncle Niels of Denmark, and was declared king in 1134.

 He punished his adversaries severely, and rewarded his supporters handsomely.

 He was killed by a subject in 1137 and was promptly succeeded by his nephew Eric III of Denmark.
</item 10>
=== END ITEM id=89cd9cff6ca4d5e4 ===

=== ITEM id=b4fdb9bc1e358e41 ===
Gold sentences and facts:
Sentence 0: 'aba], a Guarani word that means "what gives strength to the Indian") is used for the infusions of the bark of a number of trees native to Brazil.
  Fact 0.0: A Guarani word, written in the source as 'aba], means 'what gives strength to the Indian'.
  Fact 0.1: The Guarani word is used for the infusions of the bark of a number of trees native to Brazil.
Sentence 1: Guarani ( or ), specifically the primary variety known as Paraguayan Guarani (endonym "avañe'ẽ" ] 'the people's language'), is an indigenous language of South America that belongs to the Tupi–Guarani family of the Tupian languages.
  Fact 1.0: Guarani specifically refers to the primary variety known as Paraguayan Guarani.
  Fact 1.1: Paraguayan Guarani has the endonym avañe'ẽ, meaning 'the people's language'.
  Fact 1.2: Guarani is an indigenous language of South America.
  Fact 1.3: Guarani belongs to the Tupi–Guarani family of the Tupian languages.

Retrieved items:
<item 1>
Achomawi
In anthropology, Achomawi (also Achumawi, Ajumawi and Ahjumawi) are the northerly nine (out of eleven) bands of the Pit River tribe of Native Americans who
lived in what is now northeastern California in the United States.

 These nine autonomous bands (also called "tribelets") of the Pit River Indians spoke various dialects of one common language, and the other two bands spoke
dialects of a related language, called Atsugewi.

 The name "Achomawi" means river dwelling (from ajumma, "river") and the band historically inhabited the Fall River Valley and the Pit River from the south end
of Big Valley Mountains, westerly to Pit River Falls.
</item 1>

<item 2>
Catuaba
The name catuaba (pronounced [ka.two.

'aba], a Guarani word that means "what gives strength to the Indian") is used for the infusions of the bark of a number of trees native to Brazil.

 The most widely used barks are derived from the trees "Trichilia catigua" and "Erythroxylum vaccinifolium".

 Other catuaba preparations use the bark of trees from the following genera or families: "Anemopaegma", "Ilex", "Micropholis", "Phyllanthus", "Secondatia",
"Tetragastris" and species from the Myrtaceae.
</item 2>

<item 3>
Maléku language
The Maléku Jaíka language, also called Guatuso, Watuso-Wétar, and Guetar, is an indigenous language of north central Costa Rica.

 It is a Chibchan language and Votic language spoken by around 300 to 750 indigenous Maléku people.

 This language is considered to be endangered according to The Endangered Languages Project.

 Corobicí is possibly a dialect.
</item 3>

<item 4>
Cora language
Cora is an indigenous language of Mexico of the Uto-Aztecan language family.

 It is spoken by the ethnic group that is widely known as the Cora but who refer to themselves as "Naáyarite".

 The Cora inhabit the northern sierra of the Mexican state Nayarit which is named after its indigenous inhabitants.

 Cora is a Mesoamerican language and shows many of the traits defining the Mesoamerican Linguistic Area.

 Under the General Law of Linguistic Rights of the Indigenous Peoples it is recognized as a "national language" along with 62 other indigenous languages and
Spanish which have the same "validity" in Mexico .
</item 4>

<item 5>
Mactan, a densely populated Cebu island, has 470,000 residents. Sino-Tibetan is a 400-plus-language family. Sikh culture intertwines faith and Punjabi heritage.
Amka is an Israeli moshav founded by Yemenite Jews. Endangered Costa Rican Maléku and Cabécar contrast with widely spoken Paraguayan Guarani.
Indigenous-language initiatives include Australia’s Ngapartji Ngapartji project. Haisla live in British
</item 5>

<item 6>
The Umba River rises in Tanzania’s West Usambara Mountains, crosses into Kenya at its Indian Ocean mouth, and is known for gemstones and wildlife reserves.
Babati is a large Manyara district centered on Babati town. The Ovambo are Namibia’s largest ethnic group, also in Angola; the Muslim Oku descend from Yoruba
settlers in Sierra Leone and The Gambia. Ewe communities span Togo, Ghana and neighboring countries. The Gambia’s 201
</item 6>

<item 7>
Mazahua language
The Mazahua language is an indigenous language of Mexico, spoken in the country's central states by the ethnic group that is widely known as the Mazahua but
calls itself the Hñatho.

 It is a Mesoamerican language and has many of the traits of the Mesoamerican Linguistic Area.

 In 2003, along with some 62 other indigenous languages, it was recognised by a statutory law of Mexico (General Law of Linguistic Rights of the Indigenous
Peoples) as an official language in the Federal District and the other administrative divisions in which it is spoken, and on an equal footing with Spanish.

 The largest concentration of Mazahua is found in the municipality of San Felipe del Progreso, State of México, near Toluca.
</item 7>

<item 8>
Pagode (music)
Pagode is a type of Brazilian country-folk traditional style of music.

 It's also known as Cipó Preto (black liana), Pagode caipira (rural pagode) or Pagode sertanejo (folk pagode).

 Sertanejo means anything that comes from the back-country, outback or simply the countryside.

 This style of music was pioneered by musicians Tião Carreiro and Lourival dos Santos in the late 1950s, when they fused the "Coco" and "Calango de roda"
rhythms.

 Although other instruments can be added, it is much characterized by its simple, acoustic approach, with the use of just one or two viola guitars, often
accompanied by an acoustic guitar and, at times, a light percussion section.

 A structural feature commonly used in this genre is the alternation of verses sung with the accompaniement of a classical guitar, with solos of viola caipira.

 Pieces can be entirely instrumental or have one or two singers, usually: when there are two lead vocals (that can be men, women or a combination), they sing
together on a steady interval.
</item 8>

<item 9>
Cabécar language
The Cabécar language is an indigenous American language of the Chibchan language family which is spoken by the Cabécar people in Costa Rica.

 Specifically, it is spoken in the inland Turrialba Region of the Cartago Province.

 80% of speakers are monolingual; as of 2007, it is the only indigenous language in Costa Rica with monolingual adults.

 The language is also known by its dialect names Chirripó, Estrella, Telire, and Ujarrás.
</item 9>

<item 10>
Guarani language
Guarani ( or ), specifically the primary variety known as Paraguayan Guarani (endonym "avañe'ẽ" ] 'the people's language'), is an indigenous language of South
America that belongs to the Tupi–Guarani family of the Tupian languages.

 It is one of the official languages of Paraguay (along with Spanish), where it is spoken by the majority of the population, and where half of the rural
population is monolingual.

 It is spoken by communities in neighboring countries, including parts of northeastern Argentina, southeastern Bolivia and southwestern Brazil, and is a second
official language of the Argentine province of Corrientes since 2004; it is also an official language of Mercosur.
</item 10>
=== END ITEM id=b4fdb9bc1e358e41 ===

=== ITEM id=6bc887a8a2439fe1 ===
Gold sentences and facts:
Sentence 0:  He is the main vocalist of the boy group Highlight.
  Fact 0.0: The person described is the main vocalist of the boy group Highlight.
Sentence 1:  She had been a trainee at S.M. Entertainment's Starlight Academy during her middle school years before debuting as a member of the agency's girl group, Girls' Generation, in 2007.
  Fact 1.0: The person described was a trainee at S.M. Entertainment's Starlight Academy during her middle school years.
  Fact 1.1: The person described debuted as a member of Girls' Generation, the girl group of S.M. Entertainment, in 2007.
  Fact 1.2: The person described debuted in Girls' Generation after her trainee years.

Retrieved items:
<item 1>
Kim Tae-yeon discography
Apart from soundtrack recordings, Taeyeon has also recorded duets with other artists, most notably "Like a Star" with The One and "Different" with Kim Bum-soo,
which peaked at numbers one and two on the Gaon Digital Chart, respectively.
</item 1>

<item 2>
Yang Yo-seob discography
The discography of South Korean singer-songwriter, musical actor Yang Yo-seob consists of one extended play and two singles.
</item 2>

<item 3>
Kim Tae-yeon (actress)
Kim Tae-yeon (born January 3, 1976) is a South Korean actress.

 She began her entertainment career as a model, winning Model Line's 40th Fashion Model contest in 1996 and the Pantene Model contest sponsored by Ford Models
in 2000.

 Kim made her film debut in the highly controversial film "Lies" in 1999.
</item 3>

<item 4>
Kim Tae-yeon discography
Kim Tae-yeon, better known by the mononym Taeyeon, is a South Korean singer.

 Her discography consists of one studio album, two extended plays (EPs), twenty singles (including four as featured artist), and three promotional singles.

 She debuted as a member of South Korean girl group Girls' Generation in August 2007 and initially gained some popularity as a singer upon recording soundtrack
songs "If" for "Hong Gil Dong" and "Can You Hear Me" for "Beethoven Virus" (2008).

 She subsequently established herself as one of the most renowned vocalist on the South Korean music scene with further soundtrack recordings, notably "I Love
You" for "" (2010), "Missing You like Crazy" for "The King 2 Hearts", "Closer" for "To the Beautiful You" (2012), and "And One" for "That Winter, the Wind
Blows" (2013), all of which managed to enter the top ten of South Korea's Gaon Digital Chart.
</item 4>

<item 5>
Park So-yeon (singer)
Park So-yeon (born October 5, 1987), better known mononymously as Soyeon, is a South Korean singer and actress.

 She is best known as the former main vocalist and member of South Korean girl group T-ara.
</item 5>

<item 6>
The First Collage
The First Collage is the debut and first EP by Yang Yo-seob, released on November 26, 2012.
</item 6>

<item 7>
Graph query row:
person: Kim Tae-yeon
music_group: Girls' Generation
role: None
period: None
status: None
</item 7>

<item 8>
Yang Yo-seob
Yang Yo-seob (born January 5, 1990), more commonly known as Yoseob, is a South Korean singer and musical actor.

 He is the main vocalist of the boy group Highlight.
</item 8>

<item 9>
Kim Tae-yeon
Kim Tae-yeon (born March 9, 1989), referred to as Taeyeon, is a South Korean singer.

 She had been a trainee at S.M. Entertainment's Starlight Academy during her middle school years before debuting as a member of the agency's girl group, Girls'
Generation, in 2007.

 Since then, she has risen to prominence due to the group's success on the Asian music scene and further participated in the agency's projects Girls'
Generation-TTS and SM the Ballad.

 Aside from group activities, she has also recorded songs for various television dramas and movies.
</item 9>

<item 10>
Graph query row:
person: Yang Yo-seob
music_group: Highlight
role: None
period: None
status: None
</item 10>
=== END ITEM id=6bc887a8a2439fe1 ===

=== ITEM id=a70ec4e8c5c0db50 ===
Gold sentences and facts:
Sentence 0: In the middle of 1984 a Brazilian company called Prológica, which made its own versions of 8 bits US computers, brought to the Brazilian market a new equipment for its personal computer series called "CP" (shorten of Personal Computer in Portuguese).
  Fact 0.0: In the middle of 1984, a Brazilian company called Prológica brought a new equipment to the Brazilian market
  Fact 0.1: Prológica made its own versions of 8 bits US computers
  Fact 0.2: The new equipment was for Prológica's personal computer series
  Fact 0.3: The new equipment was called "CP"
  Fact 0.4: CP is short for Personal Computer in Portuguese
Sentence 1: The Atari ST is a line of home computers from Atari Corporation and the successor to the Atari 8-bit family.
  Fact 1.0: The Atari ST is a line of home computers from Atari Corporation
  Fact 1.1: The Atari ST is the successor to the Atari 8-bit family
Sentence 2:  The Atari ST is the first personal computer to come with a bitmapped color GUI, using a version of Digital Research's GEM released in February 1985.
  Fact 2.0: The Atari ST is the first personal computer to come with a bitmapped color GUI
  Fact 2.1: The Atari ST's bitmapped color GUI uses a version of Digital Research's GEM
  Fact 2.2: The version of Digital Research's GEM used by the Atari ST was released in February 1985

Retrieved items:
<item 1>
ST/Amiga Format
ST Amiga Format was a computer magazine that covered the Atari ST and Amiga computers.

 It was published by Future plc to cover the ever growing market for the, then-new, 16-bit home computers.

 Issues were equally balanced with coverage for both Commodore Amiga and Atari ST systems.

 Issue 1 included a main feature 'ST or Amiga?

 The Choice is Yours', where the pros and cons for each machine was examined.
</item 1>

<item 2>
DTACK Grounded
DTACK Grounded was a computer hobbyist newsletter published from July 1981 to September 1985 by Hal W. Hardenberg.

 Subtitled "The Journal of Simple 68000 Systems", the newsletter was dedicated to the proposition that the Motorola 68000 CPU could be used to build simple fast
computers.

 In 1981 this was a revolutionary idea.

 This was before 68000-based personal computers like the Apple Macintosh, Commodore Amiga, and Atari ST had been created.

 In 1981 Motorola was marketing the 68000 solely as a CPU for high-end Unix workstations.

 Hal believed that the 68000 could be used as a simple embedded microprocessor as well, and used the newsletter to explain how to do that.
</item 2>

<item 3>
ATASCII
The ATASCII character set, from ATARI Standard Code for Information Interchange", alternatively ATARI ASCII", is the variation on ASCII used in the Atari 8-bit
family of home computers.

 The first of this family were the Atari 400 and 800, released in 1979, and later models were released throughout the 1980s.

 The last computer to use the ATASCII character set was the Atari XEGS which was released in 1987.

 The Atari ST family of computers used the different Atari ST character set.
</item 3>

<item 4>
Atari Corporation
Atari Corporation was an American manufacturer of computers and video game consoles from 1984 to 1996.

 Atari Corp. was founded in July 1984 when Warner Communications sold the home computing and game console divisions of Atari, Inc. to Jack Tramiel.

 Its chief products were the Atari ST, , Atari 7800, Atari Lynx, and Atari Jaguar.

 The company reverse merged with JTS Inc. in 1996, becoming a small division, which itself closed when JTS liquidated the IP to Hasbro Interactive in 1998.
</item 4>

<item 5>
Atari ST
The Atari ST is a line of home computers from Atari Corporation and the successor to the Atari 8-bit family.

 The initial ST model, the 520ST, saw limited release in April-June 1985 and was widely available in July.

 The Atari ST is the first personal computer to come with a bitmapped color GUI, using a version of Digital Research's GEM released in February 1985.

 The 1040ST, released in 1986, is the first personal computer to ship with a megabyte of RAM in the base configuration and also the first with a
cost-per-kilobyte of less than US$1.
</item 5>

<item 6>
XaAES
FreeMint, MultiTOS, and a few more obscure OSes are all successors to TOS the proprietary operating system of the Atari ST computer.

 Later models of the ST Computer were called the TT and the Falcon.

 TOS stands for The Operating System or Tramiel Operating System, as Jack Tramiel was the owner of Atari.

 GEMDOS itself was a near clone of DOS, the IBM PC OS.

 This was most visible to users in the 8.3 character file naming system.

 Like all personal computer operating systems of the day, TOS was single tasking.

 GEM, the graphical user interface, was licensed from Digital Research and was not included in Apple's lawsuit against DR, thus the Macintosh like appearance
and ways of doing things remained on Atari computers.

 As time went on, the big goal for the ST within and without Atari was to have a multitasking TOS.
</item 6>

<item 7>
Megamax C
Megamax C is a K&R implementation of C for the Atari ST, Apple IIgs, and Macintosh computers.

 Sold by Megamax, Inc., based in Richardson, Texas, the package includes a one-pass compiler, linker, editor, resource construction kit, and documentation.

 In the early days of the Atari ST, Megamax C was the primary competitor to the Alcyon C compiler from Digital Research which was included in the official
developer kit from Atari.

 According to a review in "Antic" magazine, Megamax C compiled a small benchmark program six times faster than Digital Research's compiler.
</item 7>

<item 8>
Happy drives
Happy drives are series of disk drive enhancements for the Atari 8-bit and Atari ST computer families produced by a small company Happy Computers.

 Happy Computers is most noted for the add-in boards for the Atari 810 and Atari 1050 disk drives, which achieved a tremendous speed improvement for reading and
writing, and for the ability to "back up" floppies.

 Happy's products were among the most popular Atari computer add-ons.

 They were still in use and active in the aftermarket as of 2009.
</item 8>

<item 9>
Prológica CP-400
In the middle of 1984 a Brazilian company called Prológica, which made its own versions of 8 bits US computers, brought to the Brazilian market a new equipment
for its personal computer series called "CP" (shorten of Personal Computer in Portuguese).
</item 9>

<item 10>
Page 6
Page 6 (subtitled "Atari Users Magazine", and later known as New Atari User) was an independent British publication aimed at users of Atari home computers.

 It was published between 1982 and 1998.

 The computer magazine supported both the Atari 8-bit family of computers (400/800/XL/XE) and the Atari ST range.
</item 10>
=== END ITEM id=a70ec4e8c5c0db50 ===

=== ITEM id=560d13cfc96e83c2 ===
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
Chris Gorak
Chris Gorak is an American film director who started as an art director and production designer.

 He directed the 2006 film "Right at Your Door" and the 2011 film "The Darkest Hour".

 Gorak began working in the film industry in the 1990s as an art director.

 In the 2000s, he worked on several films as production designer before becoming a film director.
</item 1>

<item 2>
Consolidated Film Industries
Consolidated Film Industries was a film laboratory and film processing company and was one of the leading film laboratories in the Los Angeles area for many
decades.

 CFI processed negatives and made prints for motion pictures and television.

 The company or its employees received many Academy Awards for scientific or technical achievements.
</item 2>

<item 3>
John Biddle (yachting cinematographer)
John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.

 His films captured not only the technical aspects of sailing but also the human story in events as tranquil as a Nova Scotia cruise and as grand as the
America's Cup Races.
</item 3>

<item 4>
John Dunning (film producer)
John Dunning (April 27, 1927–September 19, 2011) was a pioneering Canadian film producer from Montreal who co-founded the Canadian film production company
Cinépix and produced early works by notable Canadian directors David Cronenberg and Ivan Reitman.

 Dunning launched Cinépix with partner André Link in Montreal in the early 1960s.

 Their biggest commercial success—and the first Canadian box office hit—came with Reitman’s "Meatballs" (1979).
</item 4>

<item 5>
Ridley Scott
Sir Ridley Scott (born 30 November 1937) is an English film director and producer.

 Following his commercial breakthrough with the science-fiction horror film "Alien" (1979), his best known works include the neo-noir dystopian science fiction
film "Blade Runner" (1982), historical drama and Best Picture Oscar winner "Gladiator" (2000), and science fiction film "The Martian" (2015).
</item 5>

<item 6>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 6>

<item 7>
Elmer Clifton
Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.

 A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work
behind the camera, with Griffith and Joseph Henabery as his mentors.

 His first feature-length solo effort as a director was "The Flame of Youth" with Jack Mulhall.
</item 7>

<item 8>
Steve McQueen (director)
Steven Rodney "Steve" McQueen (born 9 October 1969) is an English film director, producer, screenwriter, and video artist.

 For his 2013 film, "12 Years a Slave", a historical drama adaptation of an 1853 slave narrative memoir, he won an Academy Award, BAFTA Award for Best Film, and
Golden Globe Award for Best Motion Picture – Drama, as a producer, and he also received the award for Best Director from the New York Film Critics Circle.

 McQueen is the first black filmmaker to win an Academy Award for Best Picture.

 McQueen is known for his collaborations with actor Michael Fassbender, who has starred in all three of McQueen's feature films as of 2014.

 McQueen's other feature films are "Hunger" (2008), a historical drama about the 1981 Irish hunger strike, and "Shame" (2011), a drama about an executive
struggling with sex addiction.
</item 8>

<item 9>
Christopher Nolan
Christopher Edward Nolan ( ; born 30 July 1970) is an English-American film director, producer, and screenwriter.

 He is one of the highest-grossing directors in history, and among the most successful and acclaimed filmmakers of the 21st century.
</item 9>

<item 10>
Scott Free Productions
Scott Free Productions is a British film and television production company founded by filmmakers and brothers Ridley Scott and Tony Scott.

 They formed the feature film development company Percy Main Productions in 1980, naming the company after the English village Percy Main, where their father
grew up.

 The company was renamed to Scott Free Productions in 1995.

 Scott Free has produced films ranging from the 2000 Hollywood blockbuster "Gladiator" (2000) to "smaller pictures" like "Cracks" (2009).

 Between productions of "White Squall" (1996) and "G.I. Jane" (1997), Ridley Scott reorganized the company.
</item 10>
=== END ITEM id=560d13cfc96e83c2 ===

=== ITEM id=e10ee09a6c38126b ===
Gold sentences and facts:
Sentence 0: The Seven Days Battles were a series of six major battles over the seven days from June 25 to July 1, 1862, near Richmond, Virginia, during the American Civil War.
  Fact 0.0: The Seven Days Battles were a series of six major battles.
  Fact 0.1: The Seven Days Battles were fought over the seven days from June 25 to July 1, 1862.
  Fact 0.2: The Seven Days Battles were fought near Richmond, Virginia.
  Fact 0.3: The Seven Days Battles took place during the American Civil War.
Sentence 1: The Battle of Manila (February 3, 1945 – March 3, 1945) was a major battle of the Philippine campaign of 1944-45, during the Second World War.
  Fact 1.0: The Battle of Manila took place from February 3, 1945 to March 3, 1945.
  Fact 1.1: The Battle of Manila was a major battle of the Philippine campaign of 1944-45.
  Fact 1.2: The Battle of Manila took place during the Second World War.

Retrieved items:
<item 1>
Seven Days Battles
The Seven Days Battles were a series of six major battles over the seven days from June 25 to July 1, 1862, near Richmond, Virginia, during the American Civil
War.

 Confederate General Robert E. Lee drove the invading Union Army of the Potomac, commanded by Maj. Gen. George B. McClellan, away from Richmond and into a
retreat down the Virginia Peninsula.

 The series of battles is sometimes known erroneously as the Seven Days Campaign, but it was actually the culmination of the Peninsula Campaign, not a separate
campaign in its own right.
</item 1>

<item 2>
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
</item 2>

<item 3>
Seven Days Confederate order of battle
The following Confederate States Army units and commanders fought in the Seven Days Battles (from June 25 to July 1, 1862) of the American Civil War.

 Order of battle compiled from the army organization during the battle, the casualty returns and the reports.

 The Union order of battle is listed separately.
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
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 5>

<item 6>
Seven Days Union order of battle
The following Union Army units and commanders fought in the Seven Days Battles (from June 25 to July 1, 1862) of the American Civil War.

 Order of battle compiled from the army organization-return of casualties during the battle and the reports.

 The Confederate order of battle is listed separately.
</item 6>

<item 7>
Battle of Malvern Hill
The Battle of Malvern Hill, also known as the Battle of Poindexter's Farm, was fought on July 1, 1862 between the Confederate Army of Northern Virginia, led by
Gen. Robert E. Lee, and the Union Army of the Potomac under Maj. Gen. George B. McClellan.

 It was the final battle of the Seven Days Battles during the American Civil War, taking place on a 130 ft elevation of land known as Malvern Hill, near the
Confederate capital of Richmond, Virginia and just one mile (1.6 km) from the James River.

 Including inactive reserves, more than fifty thousand soldiers from each side took part, using more than two hundred pieces of artillery and three warships.
</item 7>

<item 8>
Battle of Manila (1898)
The Battle of Manila, sometimes called the Mock Battle of Manila, was a land engagement which took place in Manila on August 13, 1898, at the end of the
Spanish–American War, four months after the decisive victory by Commodore Dewey's Asiatic Squadron at the Battle of Manila Bay.

 The belligerents were Spanish forces led by Governor-General of the Philippines Fermín Jáudenes, and American forces led by United States Army Brigadier
General Wesley Merritt and United States Navy Commodore George Dewey.

 American forces were supported by units of the Philippine Revolutionary Army, led by Emilio Aguinaldo.
</item 8>

<item 9>
Battle of Manila (1945)
The Battle of Manila (February 3, 1945 – March 3, 1945) was a major battle of the Philippine campaign of 1944-45, during the Second World War.

 It was fought by American and Filipino forces against Japanese troops in Manila, the capital city of the Philippines.

 The month-long battle, which resulted in the death of over 100,000 civilians and the complete devastation of the city, was the scene of the worst urban
fighting in the Pacific theater.

 Japanese forces committed mass murder against Filipino civilians during the battle.

 Along with massive loss of life, the battle also destroyed architectural and cultural heritage dating back to the city's foundation.

 The battle ended the almost three years of Japanese military occupation in the Philippines (1942–1945).

 The city's capture was marked as General Douglas MacArthur's key to victory in the campaign of reconquest.
</item 9>

<item 10>
Battle of Guam (1944)
The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands
captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
</item 10>
=== END ITEM id=e10ee09a6c38126b ===

=== ITEM id=fd14a55d697f4213 ===
Gold sentences and facts:
Sentence 0: Lowell J. Sherman (October 11, 1885 – December 28, 1934) was an American actor and film director.
  Fact 0.0: Lowell J. Sherman was born on October 11, 1885.
  Fact 0.1: Lowell J. Sherman died on December 28, 1934.
  Fact 0.2: Lowell J. Sherman was an American actor.
  Fact 0.3: Lowell J. Sherman was an American film director.
Sentence 1: Jonathan Kaplan (born November 25, 1947) is an American film producer and director.
  Fact 1.0: Jonathan Kaplan was born on November 25, 1947.
  Fact 1.1: Jonathan Kaplan is an American film producer.
  Fact 1.2: Jonathan Kaplan is an American film director.

Retrieved items:
<item 1>
Arthur Lubin
Arthur Lubin (July 25, 1898 – May 12, 1995) was an American film director and producer who directed several "Abbott & Costello" films, "Phantom of the Opera"
(1943), the "Francis the Talking Mule" series and created the talking-horse TV series "Mister Ed".

 A prominent director for Universal Pictures in the 1940s and 1950s, he is perhaps best known today as the man who gave Clint Eastwood his first contract in
film.
</item 1>

<item 2>
D. Ross Lederman
David Ross Lederman (December 12, 1894 – August 24, 1972) was an American film director noted for his Western/action/adventure films of the 1930s and 1940s.
</item 2>

<item 3>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 3>

<item 4>
Jonathan Kaplan
Jonathan Kaplan (born November 25, 1947) is an American film producer and director.

 His film "The Accused" (1988) earned actress Jodie Foster her first Oscar for Best Actress and was nominated for the Golden Bear at the 39th Berlin
International Film Festival.

 Kaplan received five Emmy nominations for his roles directing and producing "ER".
</item 4>

<item 5>
Lowell Sherman
Lowell J. Sherman (October 11, 1885 – December 28, 1934) was an American actor and film director.

 At a time when it was highly unusual, he was both the actor and director on several films in the early 1930s, before completely transitioning to the role of
director.

 At the height of his career, after scoring huge successes with his direction of the films "She Done Him Wrong" and "Morning Glory" (which starred Mae West and
won her first Academy Award for Katharine Hepburn respectively), he succumbed to pneumonia after a brief illness.
</item 5>

<item 6>
He Knew Women
He Knew Women is a 1930 American comedy film, directed by Hugh Herbert, from a screenplay by him and William B. Jutte, which was adapted from S. N. Behrman's
1927 play "The Second Man".

 It starred Lowell Sherman and Alice Joyce, in her second to last film role.

 The film just broke even.
</item 6>

<item 7>
High Stakes (1931 film)
High Stakes is a 1931 American Pre-Code comedy drama produced and released by RKO Pictures.

 The picture was directed by Lowell Sherman who also stars and marks the last starring screen appearance of silent screen diva Mae Murray.

 It is based on a 1924 Broadway play that starred Sherman playing the same role he plays in this film.
</item 7>

<item 8>
Lawful Larceny
Lawful Larceny is a 1930 American melodramatic film, directed by Lowell Sherman from Jane Murfin's screenplay.

 The screenplay, a melodrama, was based on the play of the same name by Samuel Shipman, which originally was a comedy.

 It starred a staple of the early RKO stable, Bebe Daniels, along with Kenneth Thomson, Olive Tell and Lowell Sherman, who reprised the role he had created in
the original Broadway play.

 This film was a remake of the 1923 silent film version of the same name, produced by Famous Players-Lasky Corporation
</item 8>

<item 9>
Frederick King Keller
Frederick King Keller (born 1954 in Buffalo, New York) is an American director, producer and screenwriter for film and television.

 He is also credited under the names Frederick K. Keller, Fred K. Keller and Fred Keller.

 He is the son of actor/screenwriter .

 His father was a television pioneer who produced and directed the first weekly dramatic series seen on television.Besides acting and directing in theater his
father also ran several art house movie theaters in Buffalo which the young Fred became intimately involved with and which formed the root of his cinematic
education.
</item 9>

<item 10>
Richard Sherman (MP)
Richard Sherman ("fl."

 1364-1397) was an English ironmonger and property owner in Derby, who served two terms as a bailiff and served two terms as a Member of Parliament from Derby,
being chosen first in November 1384 (serving with John de Stockes) and again in 1391 (with Thomas Docking).
</item 10>
=== END ITEM id=fd14a55d697f4213 ===

=== ITEM id=2ec3bbe5720b142a ===
Gold sentences and facts:
Sentence 0:  He succeeded his brothers Eric IV Plovpenning and Abel of Denmark on the throne.
  Fact 0.0: The person in this sentence succeeded his brothers Eric IV Plovpenning and Abel of Denmark on the throne.
Sentence 1: Eric IV, also known as Eric Ploughpenny or Eric Plowpenny (Danish: "Erik Plovpenning" ), (  1216 – 10 August 1250) was king of Denmark from 1241 until his death in 1250.
  Fact 1.0: Eric IV is also known as Eric Ploughpenny.
  Fact 1.1: Eric IV is also known as Eric Plowpenny.
  Fact 1.2: Eric IV's Danish name is Erik Plovpenning.
  Fact 1.3: Eric IV's dates are 1216 to 10 August 1250.
  Fact 1.4: Eric IV was king of Denmark from 1241.
  Fact 1.5: Eric IV was king of Denmark until his death in 1250.

Retrieved items:
<item 1>
Prince Christian of Hanover (born 1985)
Prince Christian of Hanover (Christian Heinrich Clemens Paul Frank Peter Welf Wilhelm-Ernst Friedrich Franz Prince of Hanover, Duke of Brunswick and Lüneburg;
born 1 June 1985) is the younger son of Ernst August, Prince of Hanover, and his first wife Chantal Hochuli.

 He is the second in the line of succession to the former Hanoverian throne, after his elder brother Prince Ernst August.

 As a descendant of George III of the United Kingdom, Christian is also in the line of succession to the British throne.
</item 1>

<item 2>
Line of succession to the former Romanian throne
The succession order to the throne of the Romanian monarchy, abolished since 1947, was regulated by the monarchical constitution of 1923 and the 1884 Law of the
Romanian Royal House Rules enacted pursuant to the 1866 Constitution of Romania which had confirmed the enthronement of Prince Karl (Carol) of
Hohenzollern-Sigmaringen.

 The defunct 1923 Constitution stipulated Salic law, according to which the throne was hereditary in King Carol I's legitimate descent and, if his male issue
failed, in the descent of his brothers of the Sigmaringen princely branch of the House of Hohenzollern, according to agnatic primogeniture and to the perpetual
exclusion of females and their descendants.

 The last monarch to reign in Romania was King Michael I, who was born in 1921, abdicated his throne on 30 December 1947 under coercion, and went into exile in
Switzerland.
</item 2>

<item 3>
Eric II of Denmark
Eric II the Memorable (Danish: "Erik II Emune" ) (  1090 – 18 September 1137) was king of Denmark between 1134 and 1137.

 Eric was an illegitimate son of Eric I of Denmark, who ruled Denmark from 1095 to 1103.

 Eric the Memorable rebelled against his uncle Niels of Denmark, and was declared king in 1134.

 He punished his adversaries severely, and rewarded his supporters handsomely.

 He was killed by a subject in 1137 and was promptly succeeded by his nephew Eric III of Denmark.
</item 3>

<item 4>
Harald II of Denmark
Harald II of Denmark (died 1018) was King of Denmark from 1014 until his death in 1018.

 He was the youngest son of Sweyn Forkbeard and Gunhild of Wenden, and was regent while his father was fighting Ethelred the Unready in England.

 He inherited the Danish throne in 1014, and held it while his brother, the later king Cnut the Great conquered England.

 After his death in 1018, he was succeeded by Cnut the Great.

 Little detail is known about Harald II.
</item 4>

<item 5>
King Kang of Chu
King Kang of Chu (, died 545 BC) was from 559 to 545 BC the king of Chu, a major power during the Spring and Autumn period of ancient China.

 Born Xiong Zhao (), he succeeded his father King Gong of Chu.

 His reign was marked by constant wars with Jin, Chu's traditional enemy, and Wu, its new enemy.

 He ruled for 15 years and was succeeded by his son, Jia'ao, who would four years later be murdered by King Kang's younger brother King Ling of Chu.
</item 5>

<item 6>
Christopher I of Denmark
Christopher I (Danish: "Christoffer I" ) (1219 – 29 May 1259) was King of Denmark between 1252 and 1259.

 He was the son of Valdemar II of Denmark by his wife, Infanta Berengária of Portugal.

 He succeeded his brothers Eric IV Plovpenning and Abel of Denmark on the throne.

 Christopher was elected King upon the death of his older brother Abel in the summer of 1252.

 He was crowned at Lund Cathedral on Christmas Day 1252.
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
John Christiansen
John “Chris” Christiansen (May 1, 1923 － September 12, 1998) was the chief military test pilot for Lockheed California Company for over 30 years.

 He might be most known for having performed Lockheed S-3 Viking's maiden flight on January 21, 1972.

 His assignments also included the initial test flights of Lockheed P-3 Orion.

 Christiansen was born in Oslo, Norway in 1923 and became an American citizen in 1939.

 He later served in the U.S. Navy during World War II and the Korean Conflict.

 He began experimental flying for Lockheed Martin in 1953, and worked there until his retirement in 1984.

 He was a fellow at Society of Experimental Test Pilots.
</item 8>

<item 9>
Eric IV of Denmark
Eric IV, also known as Eric Ploughpenny or Eric Plowpenny (Danish: "Erik Plovpenning" ), (  1216 – 10 August 1250) was king of Denmark from 1241 until his death
in 1250.

 He was the son of King Valdemar II by his wife, Berengaria of Portugal, and brother of King Abel and King Christopher I.
</item 9>

<item 10>
Eric III of Denmark
Eric III Lamb (Danish: "Erik III Lam" ) (  1120 – 27 August 1146) was the King of Denmark from 1137 until 1146.

 He was the grandson of Eric I of Denmark and the nephew of Eric II of Denmark, whom he succeeded on the throne.

 He abdicated in 1146, as the first and only Danish monarch to do so.

 His succession led to a period of civil war between Sweyn III, Canute V, and Valdemar I.
</item 10>
=== END ITEM id=2ec3bbe5720b142a ===
