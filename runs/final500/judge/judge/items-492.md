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

=== ITEM id=25fc0a0cf75e4410 ===
Gold sentences and facts:
Sentence 0: Kresy Wschodnie or Kresy (] , Eastern Borderlands, or Borderlands) was a region of the Second Polish Republic during the interwar period constituting nearly half of the territory of the state; where the ethnic Poles, being the largest group, were roughly equal in their number to the size of the national minorities (with notable exceptions).
  Fact 0.0: Kresy Wschodnie, also called Kresy, Eastern Borderlands, or Borderlands, was a region of the Second Polish Republic during the interwar period.
  Fact 0.1: Kresy constituted nearly half of the territory of the Second Polish Republic.
  Fact 0.2: In Kresy, ethnic Poles were the largest group.
  Fact 0.3: In Kresy, ethnic Poles were roughly equal in number to the size of the national minorities.
  Fact 0.4: The rough equality of ethnic Poles with the national minorities in Kresy had notable exceptions.
Sentence 1:  It was drawn for the first time by the Supreme War Council as the demarcation line between the newly emerging states, the Second Polish Republic, and the Soviet Union.
  Fact 1.0: Kresy was drawn for the first time by the Supreme War Council.
  Fact 1.1: Kresy was drawn as the demarcation line between the newly emerging states, the Second Polish Republic and the Soviet Union.

Retrieved items:
<item 1>
Tripartite Indenture
The Tripartite Indenture was an agreement made in February 1405 between Owain Glyndŵr, Edmund Mortimer, and Henry Percy, 1st Earl of Northumberland, agreeing to
divide England and Wales up between them at the expense of Henry IV.

 Glyndŵr was to be given Wales, and a substantial part of the west of England, including the English portions of the Welsh Marches.

 Northumberland was to have received the north, as well as Northamptonshire, Norfolk, Warwickshire, and Leicestershire.

 The Mortimers were to have received the rest of southern England.
</item 1>

<item 2>
Register Cliff
Register Cliff is a sandstone cliff and featured key navigational landmark prominently listed in the 19th century guidebooks about the Oregon Trail, and a place
where many emigrants chiseled the names of their families on the soft stones of the cliff it was one of the key checkpoint landmarks for parties heading west
along the Platte River valley west of Fort John, Wyoming which allowed travelers to verify they were on the correct path up to South Pass and not moving into
impassable mountain terrainsgeographically, it is on the eastern ascent of the Continental divide leading upward out of the great plains in the east of the U.S.
state of Wyoming.

 It is notable as a historic landmark for 'registering' hundreds of emigrants on the Oregon Trail (thus also the other northern Emigrant Trails that split off
farther west such as the California Trail and Mormon Trail) who came to follow custom and inscribed their names on its rocks during the western migrations of
the 19th century.

 An estimated 500,000 emigrants used these trails from 1843–1869, with up to one-tenth dying along the way, usually due to disease.
</item 2>

<item 3>
New York – New Jersey Line War
The New York – New Jersey Line War (also known as the N.J. Line War) refers to a series of skirmishes and raids that took place for over half a century between
1701 and 1765 at the disputed border between two American colonies, the Province of New York and the Province of New Jersey.
</item 3>

<item 4>
Curzon Line
The history of the Curzon Line, with minor variations, goes back to the period following World War I.

 It was drawn for the first time by the Supreme War Council as the demarcation line between the newly emerging states, the Second Polish Republic, and the
Soviet Union.

 The proposal was put forward by British Foreign Secretary George Curzon, to serve as a diplomatic basis for the future border agreement, and in that form, it
never materialized because the war went on.
</item 4>

<item 5>
Kresy
In September 1939, after the Soviet Union joined Nazi Germany in their attack on Poland in accordance with the Molotov–Ribbentrop Pact, the territories were
incorporated into Soviet Ukraine, Belarus and Lithuania in the atmosphere of terror.
</item 5>

<item 6>
Kresy
Kresy Wschodnie or Kresy (] , Eastern Borderlands, or Borderlands) was a region of the Second Polish Republic during the interwar period constituting nearly
half of the territory of the state; where the ethnic Poles, being the largest group, were roughly equal in their number to the size of the national minorities
(with notable exceptions).

 Administratively, the territory of Kresy was composed of voivodeships of Lwów, Nowogródek, Polesie, Stanisławów, Tarnopol, Wilno, Wołyń, and the Białystok.

 Today, these territories are divided between Western Ukraine, Western Belarus, and south-eastern Lithuania, with such major cities as Lviv, Vilnius, and Grodno
no longer in Poland.

 In the Second Polish Republic the term "Kresy" roughly equated with the lands beyond the so-called Curzon Line, which was suggested after World War I in
December 1919 by the British Foreign Office as the eastern border of the re-emerging sovereign Republic following the century of partitions.
</item 6>

<item 7>
Lac La Belle and Calumet Railroad
The Lac La Belle and Calumet Railroad was an American, narrow gauge railroad that operated in the Keweenaw Peninsula, or the extreme northern Upper Peninsula of
Michigan.

 The line ran between a stamp mill at Lac La Belle and two copper mines, the Mendota and the Delaware, from 1883 to 1888, when poor economic conditions forced
the line's closure.

 The defunct company and its tracks were later purchased by the Copper Range Company, which converted the line to standard gauge and changed the railroad's name
to the Keweenaw Central.

 This company failed in 1917, and the former Lac La Belle and Calumet tracks were removed in 1918.
</item 7>

<item 8>
William Pynchon
Pynchon's stance led to Springfield aligning with the faraway government of the Massachusetts Bay Colony rather than the more geographically and ideologically
compatible Connecticut Colony.
</item 8>

<item 9>
Anti-Nebraska movement
The Anti-Nebraska movement was a political alignment in the United States formed in opposition to the Kansas-Nebraska Act of 1854 and to its repeal of the
Missouri Compromise provision forbidding slavery in U.S. territories north of latitude 36° 30' N.

 (At the time, the name "Nebraska" could loosely refer to areas west of the Missouri River.)
</item 9>

<item 10>
Battle of Berestechko
The Battle of Berestechko (Polish: "Bitwa pod Beresteczkiem" ; Ukrainian: Берестецька битва, Битва під Берестечком ) was fought between the Ukrainian Cossacks,
led by Hetman Bohdan Khmelnytsky, aided by their Crimean Tatar allies, and a Polish army under King John II Casimir.

 It was a battle of a Cossack rebellion in Ukraine that took place in the years 1648–1657 after the expiration of a two-year truce.

 Fought from 28 to 30 June 1651, the battle took place in the province of Volhynia, on the hilly plain south of the Styr River.

 The Polish camp was on the river opposite Berestechko and faced south, towards the Cossack army about two kilometers away, whose right flank was against the
River Pliashivka (Pliashova) and the Tatar army on their left flank.

 It was probably the largest European land battle of the 17th century.
</item 10>
=== END ITEM id=25fc0a0cf75e4410 ===

=== ITEM id=effadb57a870f5cf ===
Gold sentences and facts:
Sentence 0: The Korean War (in South Korean Hangul: 한국전쟁 ; Hanja: 韓國戰爭 ; RR: "Hanguk Jeonjaeng " , "Korean War"; in North Korean Chosŏn'gŭl: 조국해방전쟁 ; Hancha: 祖國解放戰爭 ; MR: "Choguk haebang chǒnjaeng " , "Fatherland Liberation War"; 25 June 1950 – 27 July 1953) was a war between North Korea (with the support of China and the Soviet Union) and South Korea (with the principal support of the United States).
  Fact 0.0: The Korean War is called 한국전쟁 in South Korean Hangul.
  Fact 0.1: The Korean War is called 韓國戰爭 in South Korean Hanja.
  Fact 0.2: The Korean War is romanized as 'Hanguk Jeonjaeng' (RR), meaning 'Korean War'.
  Fact 0.3: The Korean War is called 조국해방전쟁 in North Korean Chosŏn'gŭl.
  Fact 0.4: The Korean War is called 祖國解放戰爭 in North Korean Hancha.
  Fact 0.5: The Korean War is romanized as 'Choguk haebang chǒnjaeng' (MR), meaning 'Fatherland Liberation War'.
  Fact 0.6: The Korean War lasted from 25 June 1950 to 27 July 1953.
  Fact 0.7: The Korean War was a war between North Korea and South Korea.
  Fact 0.8: North Korea fought with the support of China and the Soviet Union.
  Fact 0.9: South Korea fought with the principal support of the United States.
Sentence 1: The Western Allied invasion of Germany was coordinated by the Western Allies during the final months of hostilities in the European theatre of World War II.
  Fact 1.0: The Western Allied invasion of Germany was coordinated by the Western Allies.
  Fact 1.1: The Western Allied invasion of Germany took place during the final months of hostilities in the European theatre of World War II.
Sentence 2:  The Allied invasion of Germany started with the Western Allies crossing the Rhine River in March 1945 before fanning out and overrunning all of western Germany from the Baltic in the north to Austria in the south before the Germans surrendered on 8 May 1945.
  Fact 2.0: The Allied invasion of Germany started with the Western Allies crossing the Rhine River in March 1945.
  Fact 2.1: After crossing the Rhine, the Western Allies fanned out and overran all of western Germany.
  Fact 2.2: The Western Allies overran western Germany from the Baltic in the north to Austria in the south.
  Fact 2.3: The Germans surrendered on 8 May 1945.

Retrieved items:
<item 1>
Western Allied invasion of Germany
The Western Allied invasion of Germany was coordinated by the Western Allies during the final months of hostilities in the European theatre of World War II.

 The Allied invasion of Germany started with the Western Allies crossing the Rhine River in March 1945 before fanning out and overrunning all of western Germany
from the Baltic in the north to Austria in the south before the Germans surrendered on 8 May 1945.

 This is known as the "Central Europe Campaign" in United States military histories.
</item 1>

<item 2>
Operation Epsom
Operation Epsom, also known as the First Battle of the Odon, was a British Second World War offensive that took place between 26 and 30 June 1944, during the
Battle of Normandy.

 The offensive was intended to outflank and seize the German-occupied city of Caen, an important Allied objective, in the early stages of Operation Overlord,
the Allied invasion of north-west Europe.
</item 2>

<item 3>
Schlieffen Plan
The Schlieffen Plan (German: "Schlieffen-Plan" , ] ) was the name given after World War I to the thinking behind the German invasion of France and Belgium on 4
August 1914.

 Field Marshal Alfred von Schlieffen, the Chief of the Imperial Army German General Staff from 1891 to 1906, devised a deployment plan for a war-winning
offensive, in a one-front war against the French Third Republic from 1905–06.

 After the war, the German official historians of the "Reichsarchiv" and other writers, described the plan as a blueprint for victory.

 German historians claimed that the plan had been ruined by "Generaloberst" (Colonel-General) Helmuth von Moltke the Younger, the Commander-in-Chief of the
German army after Schlieffen retired in 1906, who was dismissed after the First Battle of the Marne (5–12 September 1914).
</item 3>

<item 4>
The entries cover Allied operations across World War II and one ancient battle: the 1943 invasion of Italy (Salerno, Calabria, Taranto); Normandy’s 1944 Epsom
offensive and Gold Beach landings; the 21st Army Group’s campaign in northwest Europe; and the 1945 Allied advance into Germany and southern Germany. Other
events include Operation Barras’s 2000 rescue of British hostages in Sierra Leone, the deadly 1943 Naples post-office bombing
</item 4>

<item 5>
Korean War
The Korean War (in South Korean Hangul: 한국전쟁 ; Hanja: 韓國戰爭 ; RR: "Hanguk Jeonjaeng " , "Korean War"; in North Korean Chosŏn'gŭl: 조국해방전쟁 ; Hancha: 祖國解放戰爭 ; MR:
"Choguk haebang chǒnjaeng " , "Fatherland Liberation War"; 25 June 1950 – 27 July 1953) was a war between North Korea (with the support of China and the Soviet
Union) and South Korea (with the principal support of the United States).

 The war began on 25 June 1950 when North Korea invaded South Korea.

 The United Nations, with the United States as the principal force, came to the aid of South Korea.

 China came to the aid of North Korea, and the Soviet Union also gave some assistance to the North.
</item 5>

<item 6>
World War II caused an estimated 50–80+ million deaths: 50–55 million civilians and 21–25 million military personnel. Major Pacific battles included Bataan
(1942; mass surrender and Death March), Attu (1943), Saipan (1944), Manila (1945; over 100,000 civilians killed), Guam (1941 defeat, 1944 recapture), and
Okinawa (1945; 82 days). Other conflicts covered include
</item 6>

<item 7>
Battle of the Ch'ongch'on River
The Battle of the Ch'ongch'on River, also known as the Battle of the Ch'ongch'on () or the Second Phase Campaign Western Sector (), was a decisive battle in the
Korean War, and it took place from November 25 to December 2, 1950, along the Ch'ongch'on River Valley in the northwestern part of North Korea.

 In response to the successful Chinese First Phase Campaign against the United Nations forces, General Douglas MacArthur launched the Home-by-Christmas
Offensive to evict the Chinese forces from Korea and to end the war.

 Anticipating this reaction, the Chinese People's Volunteer Army Commander Peng Dehuai planned a counteroffensive, dubbed the "Second Phase Campaign", against
the advancing UN forces.
</item 7>

<item 8>
Allied invasion of Italy
The Allied invasion of Italy was the Allied amphibious landing on mainland Italy that took place on 3 September 1943 during the early stages of the Italian
Campaign of World War II.

 The operation was undertaken by General Sir Harold Alexander's 15th Army Group (comprising General Mark W. Clark and General Bernard Montgomery's British
Eighth Army) and followed the successful invasion of Sicily.

 The main invasion force landed around Salerno on 9 September on the western coast in Operation Avalanche, while two supporting operations took place in
Calabria (Operation Baytown) and Taranto (Operation Slapstick).
</item 8>

<item 9>
South German Offensive
The South German Offensive is the general name of one of the final offensives of World War II in Europe.

 The offensive was led by the Seventh and Third armies of the United States along with the First Army of France.

 Soviet troops linked up with American forces in Czechoslovakia notably in the Battle of Slivice.

 The offensive was made by the US 6th Army Group to protect the US 12th Army Group's right flank and to prevent a German last stand in the Alps.

 However German resistance was much more fierce than in the north, which slowed the 6th Army Group's progress.

 However, by the end of April, many German divisions surrendered without a fight to the advancing American forces to avoid the inevitable destruction.

 The VI Corps of the Seventh Army linked up with the US Fifth Army, which fought through Italy, in the Alps as the Third Army advanced into Austria and
Czechoslovakia, where it linked up with Soviet forces advancing from the east.

 Fighting continued a few days after the Surrender of Germany on 8 May, due to German forces fighting west to surrender to the Americans instead of the Soviets.
</item 9>

<item 10>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 10>
=== END ITEM id=effadb57a870f5cf ===

=== ITEM id=9d5532bae21a9001 ===
Gold sentences and facts:
Sentence 0: During World War II, Operation Cold Comfort was a failed SAS raid that began with a parachute drop north of Verona on February 17, 1945.
  Fact 0.0: Operation Cold Comfort was a failed SAS raid.
  Fact 0.1: Operation Cold Comfort took place during World War II.
  Fact 0.2: Operation Cold Comfort began with a parachute drop.
  Fact 0.3: The parachute drop of Operation Cold Comfort was north of Verona.
  Fact 0.4: The parachute drop of Operation Cold Comfort was on February 17, 1945.
Sentence 1:  The SAS was founded in 1941 as a regiment, and later reconstituted as a corps in 1950.
  Fact 1.0: The SAS was founded in 1941 as a regiment.
  Fact 1.1: The SAS was later reconstituted as a corps.
  Fact 1.2: The SAS was reconstituted as a corps in 1950.

Retrieved items:
<item 1>
Joint Special Operations Command (Jordan)
The Special Operation Forces of the Jordanian Armed Forces serve as Jordan's premiere special forces unit.

 Founded on April 15, 1963 on the orders of the late King Hussein, its primary roles include reconnaissance, counter-terrorism, search and evacuation,
intelligence gathering combat, and the protection of key sites.

 The Special Operation Forces are also charged with carrying out precision strikes against critical enemy targets.

 The 14,000-strong unit are equipped and trained to be able to operate behind enemy lines for long periods without any logistical support, and is considered one
of the finest special forces units in the world.
</item 1>

<item 2>
Operation Lighthouse (1937)
Operation Lighthouse was the name given to the failed experimental launch of four Aggregate 3 liquid-fuel rockets by Wernher von Braun and Walter Dornberger on
the German island of Greifswalder Oie in December 1937.
</item 2>

<item 3>
Special Operations Force (Singapore)
The Special Operations Force (SOF) is the Republic's Army Special Forces composed of highly trained elite soldiers within the Singapore Armed Forces Commando
Formation and an essential component of the joint special forces unit Special Operations Task Force (SOTF).

 According to the Principles of Special Forces, the Republic's special forces thrive on their exceptional qualities and advanced skills, consisting highly
adaptive individuals who are independent and can operate independently, and the Republic's special forces soldiers cannot be mass-produced and must be managed
carefully.
</item 3>

<item 4>
Mark E. Mitchell
Mark E. Mitchell is the current Acting Assistant Secretary of Defense for Special Operations and Low-Intensity Conflict of the United States Department of
Defense.

 A retired Colonel, Mitchell was the first member of the United States Army to be awarded the Distinguished Service Cross during the War in Afghanistan and was
the first to receive the award since the Vietnam War.

 In 2003, he received the award for his actions during the Battle of Qala-i-Jangi, which took place in late November – early December 2001.
</item 4>

<item 5>
Operation Cold Comfort
During World War II, Operation Cold Comfort was a failed SAS raid that began with a parachute drop north of Verona on February 17, 1945.

 It was later renamed Zombie.
</item 5>

<item 6>
Jan Breytenbach
Jan Dirk Breytenbach {'1': ", '2': ", '3': ", '4': "} (born 4 July 1933) was appointed by General Fritz Loots, the founder of the South African Special Forces
Brigade, as the first commander of 1 Reconnaissance Commando, the first unit founded within the South African Special Forces.

 He was also appointed as the first commander of the 32 Battalion, known colloquially as "Buffalo Battalion", as well as 44 Parachute Brigade.
</item 6>

<item 7>
Battle of Prek Klok II
The Battle of Prek Klok II occurred on March 10, 1967, during Operation Junction City when American military forces were conducting a search and destroy
operation against the Viet Cong forces in Tay Ninh Province west of the capital of South Vietnam, Saigon.

 During the course of the operation they had already had a significant engagement in the Battle of Prek Klok I.

 During the night, Artillery Fire Support Patrol Base II at Prek Klok was attacked by two communist battalions, resulting in a short battle.

 This was the second major battle of Operation Junction City.

 The communists started by mortaring the base and launching anti-tank fire at the armored personnel carriers (APCs) surrounding the base.

 Attacks came from the north and east, followed by an infantry charge out of wooded areas from the southwest.

 With the help of air strikes from nearby planes, as well as artillery and ample supplies flown in by helicopter, the Americans easily repelled the communist
attack, which consisted of two battalions.

 The Americans killed 197 communists but lost only three of their men.
</item 7>

<item 8>
Operation Diadem order of battle
Operation Diadem order of battle is a listing of the significant formations that were involved in the fighting on the Winter Line and at the Anzio bridgehead
south of Rome during "Operation Diadem" in May - June 1944 which resulted in the Allied breakthrough at Cassino and the breakout at Anzio leading to the capture
of Rome.
</item 8>

<item 9>
Special Air Service
The Special Air Service (SAS) is a special forces unit of the British Army.

 The SAS was founded in 1941 as a regiment, and later reconstituted as a corps in 1950.

 The unit undertakes a number of roles including covert reconnaissance, counter-terrorism, direct action and hostage rescue.
</item 9>

<item 10>
Operation Barras
Operation Barras was a British Army operation that took place in Sierra Leone on 10 September 2000, during the late stages of that nation's civil war.

 The operation aimed to release five British soldiers of the Royal Irish Regiment who had been held by a militia group known as the "West Side Boys".

 The soldiers were part of a patrol that was returning from a visit to Jordanian peacekeepers attached to the United Nations Mission in Sierra Leone (UNAMSIL)
at Masiaka on 25 August 2000 when they turned off the main road and down a track towards the village of Magbeni.

 There the patrol was overwhelmed by a large number of heavily armed rebels, taken prisoner, and transported to Gberi Bana on the opposite side of Rokel Creek.
</item 10>
=== END ITEM id=9d5532bae21a9001 ===

=== ITEM id=0f9c5c8089572209 ===
Gold sentences and facts:
Sentence 0: Willi Forst, born Wilhelm Anton Frohs (7 April 1903 – 11 August 1980) was an Austrian actor, screenwriter, film director, film producer and singer.
  Fact 0.0: Willi Forst was born Wilhelm Anton Frohs.
  Fact 0.1: Willi Forst's dates are 7 April 1903 to 11 August 1980.
  Fact 0.2: Willi Forst was an Austrian actor.
  Fact 0.3: Willi Forst was a screenwriter.
  Fact 0.4: Willi Forst was a film director.
  Fact 0.5: Willi Forst was a film producer.
  Fact 0.6: Willi Forst was a singer.
Sentence 1: Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.
  Fact 1.0: Elmer Clifton's dates are March 14, 1890 to October 15, 1949.
  Fact 1.1: Elmer Clifton was an American writer.
  Fact 1.2: Elmer Clifton was an American director.
  Fact 1.3: Elmer Clifton was an American actor.
  Fact 1.4: Elmer Clifton was active from the early silent days.

Retrieved items:
<item 1>
Elmer Clifton
Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.

 A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work
behind the camera, with Griffith and Joseph Henabery as his mentors.

 His first feature-length solo effort as a director was "The Flame of Youth" with Jack Mulhall.
</item 1>

<item 2>
Arthur Lubin
Arthur Lubin (July 25, 1898 – May 12, 1995) was an American film director and producer who directed several "Abbott & Costello" films, "Phantom of the Opera"
(1943), the "Francis the Talking Mule" series and created the talking-horse TV series "Mister Ed".

 A prominent director for Universal Pictures in the 1940s and 1950s, he is perhaps best known today as the man who gave Clint Eastwood his first contract in
film.
</item 2>

<item 3>
Kaiserjäger (film)
Kaiserjäger is a 1956 Austrian film directed by Willi Forst.
</item 3>

<item 4>
Ernst Messerschmid
Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
</item 4>

<item 5>
Alf Clausen
Alf Heiberg Clausen (born March 28, 1941) is an American film and television composer.

 He is best known for his work scoring many episodes of "The Simpsons", of which he had been the sole composer between 1990 and 2017.

 Clausen has scored or orchestrated music for more than 30 films and television shows, including "Moonlighting", "The Naked Gun", "ALF" and "Ferris Bueller's
Day Off".
</item 5>

<item 6>
Witmer Stone
Witmer Stone (September 22, 1866 – May 24, 1939) was an American ornithologist, botanist, and mammalogist, and was considered one of the last of the “great
naturalists.”

 Stone is remembered principally as an ornithologist.

 He was president of the American Ornithologists’ Union (AOU) 1920–23, and was editor of the AOU’s periodical "The Auk" 1912–1936.

 He spearheaded the production of the 4th edition of the AOU checklist, published in 1931.

 He worked for over 50 years in the Ornithology Department at the Academy of Natural Sciences of Philadelphia, eventually serving as Director of the
institution.

 Stone was one of the founding members of the Delaware Valley Ornithological Club (DVOC) in 1890 and was actively involved in the organization for the remainder
of his life.

 Stone was one of only two scientists (Joseph Grinnell was the other) to serve as president of both the AOU and the American Society of Mammalogists, and he
co-authored two popular books about mammals.
</item 6>

<item 7>
Willi Forst
Willi Forst, born Wilhelm Anton Frohs (7 April 1903 – 11 August 1980) was an Austrian actor, screenwriter, film director, film producer and singer.

 As a debonair actor he was a darling of the German-speaking film audiences, as a director, one of the most significant makers of the Viennese period musical
melodramas and comedies of the 1930s known as "Wiener Filme".

 From the mid-1930s he also recorded many records, largely of sentimental Viennese songs, for the Odeon Records label owned by Carl Lindström AG.
</item 7>

<item 8>
Everett Greenbaum
Everett Greenbaum (December 20, 1919 – July 11, 1999) was an American television and film writer and actor who contributed to such shows as "The Andy Griffith
Show" (24 Episodes), "M*A*S*H" (35 Episodes), "Love American Style", "The Real McCoys" (32 Episodes), "Sanford and Son", and "The George Gobel Show".

 Greenbaum was a co-creator with Jim Fritzell of "Mister Peepers" an important early television show which starred Wally Cox.

 He wrote the Hollywood feature film "Good Neighbor Sam", as well as a series of films starring Don Knotts that included "The Shakiest Gun in the West", "The
Reluctant Astronaut", and "The Ghost and Mr. Chicken".
</item 8>

<item 9>
Winfred Overholser
Winfred Overholser (1892 – October 6, 1964) was an American psychiatrist, president of the American Psychiatric Association, and for 25 years the superintendent
of St. Elizabeths Hospital, a federal institution for the mentally ill in Washington, D.C.
</item 9>

<item 10>
Walter Dornberger
Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was a German Army artillery officer whose career spanned World War I and World War
II.

 He was a leader of Nazi Germany's V-2 rocket program and other projects at the Peenemünde Army Research Center.
</item 10>
=== END ITEM id=0f9c5c8089572209 ===

=== ITEM id=f4e9023f9b440680 ===
Gold sentences and facts:
Sentence 0: Race to Nowhere is a 2009 documentary film written by Maimone Attia and directed by Vicki Abeles and Jessica Congdon.
  Fact 0.0: Race to Nowhere is a 2009 documentary film.
  Fact 0.1: Race to Nowhere was written by Maimone Attia.
  Fact 0.2: Race to Nowhere was directed by Vicki Abeles.
  Fact 0.3: Race to Nowhere was directed by Jessica Congdon.
Sentence 1: SlamNation is a documentary film by director Paul Devlin.
  Fact 1.0: SlamNation is a documentary film.
  Fact 1.1: SlamNation was directed by Paul Devlin.
Sentence 2:  The film follows the National Poetry Slam in Portland, Oregon.
  Fact 2.0: SlamNation follows the National Poetry Slam.
  Fact 2.1: The National Poetry Slam is in Portland, Oregon.

Retrieved items:
<item 1>
The entries cover documentaries on Boston hardcore (All Ages, 2012), punk (Punk’s Not Dead, 2007; American Hardcore, 2006), bipolar disorder, artistic aging,
slavery, archaeology, bookstores, and Charles Bukowski; plus Barbara Hammer’s autobiographical Tender Fictions. Fiction films include Bakshi’s acclaimed X-rated
Heavy Traffic, Heavyweights, several American Pie and Bring It On sequels, Schrader’s Hardcore, Blue Collar, and The
</item 1>

<item 2>
Poetry in Motion (film)
Poetry in Motion is a 1982 Canadian documentary film directed by Ron Mann featuring contemporary North American poetry and music.

 Featured are some of the Black Mountain poets, Beats, minimalist poets, and avant-garde poets.

 It was released in theaters, later being distributed on VHS, LaserDisc, and DVD.

 An extended CD-ROM version was also released.
</item 2>

<item 3>
Rick Ray
Rick Ray is an American filmmaker best known for his 2006 documentary film 10 Questions for the Dalai Lama which he wrote, filmed and directed.

 His documentary "Lynching Charlie Lynch", about the trials of former medical marijuana dispensary owner Charles C. Lynch, premiered at the 2011 San Luis Obispo
International Film Festival on March 9.
</item 3>

<item 4>
Bukowski (1973 film)
Bukowski is a 1973 documentary film produced by Taylor Hackford and directed by Richard Davies that follows Los Angeles poet Charles Bukowski to San Francisco
for a poetry reading.

 The full 60-minute documentary begins with footage of Bukowski in his Los Angeles home and neighborhood as he discusses his history as a postal worker as well
as his approach to and perspective on poetry.

 The film then shows him flying with Linda King to San Francisco for the poetry reading followed by interactions with attendees after the show.

 One night the window of his room is broken during a fight between some guests and then a fight between Charles and Linda causes her to leave.

 Interviews follow with Liza and Linda about their relationship with Charles.

 Bukowski is shown betting at the track and explaining his betting strategy.
</item 4>

<item 5>
Race to Nowhere
Race to Nowhere is a 2009 documentary film written by Maimone Attia and directed by Vicki Abeles and Jessica Congdon.
</item 5>

<item 6>
The entries survey documentaries, films, TV, and film-related concepts: Los Angeles hip-hop, Aboriginal country music, Latin American politics, oceans, nerd
culture, Iraqi life and war, Bougainville, education, disability, poetry, surfing, and family relationships. Featured works include *Gasland*, *Baghdad ER*,
*War/Dance*, *First Life*, and *The Coconut Revolution*. Several received major awards or nominations. The material also profiles filmmaker Wayne Coles-Janess
and defines
</item 6>

<item 7>
Beyond This Place (2010 film)
Beyond This Place (2010) is a documentary film directed by Kaleo La Belle, screened at a number of film festivals.

 In it, La Belle reunites with his biological father Cloud Rock La Belle, a charismatic figure who was largely absent from his life.

 The film deals, in a personal and intimate way, with issues of parenting, issues of freedom versus responsibility, and with the aging of the 60's generation.
</item 7>

<item 8>
We Are Poets
We Are Poets is a documentary film directed by Daniel Lucchesi and Alex Ramseyer-Bache.

 It follows six young poets from Leeds Young Authors performance poetry group in Leeds on their visit to the international Poetry Slam Competition in Washington
D.C, United States.

 It premièred at Europe's largest Documentary Film festival - Doc/Fest in Sheffield's Showroom Cinema on 11 June 2011, where it won the coveted Sheffield Youth
Jury Award.
</item 8>

<item 9>
SlamNation
SlamNation is a documentary film by director Paul Devlin.

 The film follows the National Poetry Slam in Portland, Oregon.
</item 9>

<item 10>
War/Dance
War/Dance is a 2007 American documentary film written and directed by Sean Fine and Andrea Nix Fine and produced by Shine Global's Susan MacLaury, a professor
at Kean University, and Albie Hecht.

 It was nominated for the 2008 Academy Award for Best Documentary Feature and received the Emmy Award for Best Documentary and Best Cinematography in 2010.
</item 10>
=== END ITEM id=f4e9023f9b440680 ===

=== ITEM id=067e7ba4fe32d477 ===
Gold sentences and facts:
Sentence 0: Liu Ailing (; born June 2, 1967) is a Chinese former footballer who played for the China national team at the 1991, 1995 and 1999 editions of the FIFA Women's World Cup.
  Fact 0.0: Liu Ailing was born on June 2, 1967.
  Fact 0.1: Liu Ailing is a Chinese former footballer.
  Fact 0.2: Liu Ailing played for the China national team at the 1991 FIFA Women's World Cup.
  Fact 0.3: Liu Ailing played for the China national team at the 1995 FIFA Women's World Cup.
  Fact 0.4: Liu Ailing played for the China national team at the 1999 FIFA Women's World Cup.
Sentence 1: The FIFA Women's World Cup is an international football competition contested by the senior women's national teams of the members of "Fédération Internationale de Football Association" (FIFA), the sport's global governing body.
  Fact 1.0: The FIFA Women's World Cup is an international football competition.
  Fact 1.1: The FIFA Women's World Cup is contested by the senior women's national teams of the members of FIFA.
  Fact 1.2: FIFA stands for Fédération Internationale de Football Association.
  Fact 1.3: FIFA is the sport's global governing body.

Retrieved items:
<item 1>
Australia women's national soccer team
The Australian women's national soccer team represents Australia in international women's soccer at the senior level.

 The team is overseen by the governing body for soccer in Australia, Football Federation Australia (FFA), which is currently a member of the Asian Football
Confederation (AFC) and the regional ASEAN Football Federation (AFF) since leaving the Oceania Football Confederation (OFC) in 2006.

 The team's official nickname is the Matildas (from the Australian folk song "Waltzing Matilda"), having been known as the Female Socceroos before 1995.

 The current team manager is Alen Stajcic.
</item 1>

<item 2>
Women's football at the 2015 Pacific Games
The 2015 Pacific Games women's football tournament was the 4th edition of the Pacific Games women's football tournament.

 The women's football tournament was held in Port Moresby, Papua New Guinea between 6–16 July 2015 as part of the 2015 Pacific Games.

 The tournament was open to full women's national teams (unlike the men's tournament, which was age-restricted).
</item 2>

<item 3>
EAFF E-1 Football Championship (women)
EAFF E-1 Football Championship is an international football competition in East Asia for national teams of the East Asian Football Federation (EAFF).

 The competition between women's national teams is held alongside men's competition.
</item 3>

<item 4>
Women's Nordic Football Championship
Women's Nordic Football Championship was an international football competition contested by the women's national football teams of the Nordic countries.

 The tournament was held annually between 1974 and 1982.

 Finland, Denmark and Sweden competed from the start, Norway joined the tournament in 1978.

 Iceland and Faroe Islands did not take part at the competition.
</item 4>

<item 5>
Geri Donnelly
Geraldine "Geri" Donnelly (born 30 November 1965) is a Canadian former soccer player.

 A midfielder, she represented Canada at the 1995 and 1999 editions of the FIFA Women's World Cup and was named Canadian Player of the Year in 1996 and 1999.

 Donnelly was part of the Canadian squad who won the 1998 CONCACAF Women's Championship.

 She was selected as a member of the Canadian Soccer Hall of Fame in 2014.
</item 5>

<item 6>
Liu Ailing
Liu Ailing (; born June 2, 1967) is a Chinese former footballer who played for the China national team at the 1991, 1995 and 1999 editions of the FIFA Women's
World Cup.

 She won a silver medal at the 1996 Atlanta Olympics and participated at the 2000 Sydney Olympics.

 A playmaking midfielder, she played professional club football in Japan and the United States.
</item 6>

<item 7>
Cassie Law
Cassandra (Cassie) Law is a former American women's rugby union player and was a member of the 1991 World Cup Championship team and the 1994 Women's World Cup
Team.

 She began her rugby career in 1985 while attending the University of Oregon.

 Post graduation in 1988, she moved to San Francisco, California to begin her federal career with the US Department of Veterans Affairs.

 Her rugby career continued in California playing for the Bay Area SheHawks (BASH) where she competed from 1988 - 1995 winning 3 National Titles.

 During these years, she represented the Pacific Coast Grizzlies territorial team and was selected to join the USA Eagles Women's National Rugby Team competing
from 1990 - 1994.

 Her competitive rugby career as a United States Eagle began at the 1990 World Cup Festival in New Zealand and continued through the 1991 Inaugural Women's
Rugby World Cup Championship in Cardiff, Wales where the United States defeated England in the Final to take gold and the 1994 Women's Rugby World Cup in
Edinburgh, Scotland where the Eagles fell to England taking second place.
</item 7>

<item 8>
EAFF E-1 Football Championship
EAFF E-1 Football Championship, known as the East Asian Football Championship from 2003 to 2010, and the EAFF East Asian Cup for the 2013 and 2015 editions, is
a men's international football competition in East Asia for member nations of the East Asian Football Federation (EAFF).

 Before the EAFF was founded in 2002, the Dynasty Cup was held between the East Asian top four teams, and was regarded as the unofficial East Asian
Championship.

 There is a separate competition for both men (first held in 2003) and women (first held in 2005).

 There was also a combined points competition in 2005, where the results of the men's and women's teams are added together (not including qualifiers).
</item 8>

<item 9>
Gambia women's national football team
The Gambia women's national football team represents the Gambia in international football competition.

 The team, however, has not competed in a match recognised by FIFA, the sport's international governing body, despite that organised women's football has been
played in the country since 1998.

 The Gambia has two youth teams, an under-17 side that has competed in FIFA U-17 Women's World Cup qualifiers, and an under-19 side that withdrew from regional
qualifiers for an under-19 World Cup.

 The development of a national team faces challenges similar to those across Africa, although the national football association has four staff members focusing
on women's football.
</item 9>

<item 10>
FIFA Women's World Cup
The FIFA Women's World Cup is an international football competition contested by the senior women's national teams of the members of "Fédération Internationale
de Football Association" (FIFA), the sport's global governing body.

 The competition has been held every four years since 1991, when the inaugural tournament, then called the Women's World Championship, was held in China.
</item 10>
=== END ITEM id=067e7ba4fe32d477 ===

=== ITEM id=f4e003f533444938 ===
Gold sentences and facts:
Sentence 0: Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.
  Fact 0.0: Remember the Titans is a 2000 American sports film
  Fact 0.1: Remember the Titans was produced by Jerry Bruckheimer
  Fact 0.2: Remember the Titans was directed by Boaz Yakin
Sentence 1: My Favorite Martian is a 1999 American science fiction comedy film starring Christopher Lloyd, Jeff Daniels, Daryl Hannah, Elizabeth Hurley, Wallace Shawn and Ray Walston, based on the 1960s television series of the same name in which Walston starred.
  Fact 1.0: My Favorite Martian is a 1999 American science fiction comedy film
  Fact 1.1: My Favorite Martian stars Christopher Lloyd
  Fact 1.2: My Favorite Martian stars Jeff Daniels
  Fact 1.3: My Favorite Martian stars Daryl Hannah
  Fact 1.4: My Favorite Martian stars Elizabeth Hurley
  Fact 1.5: My Favorite Martian stars Wallace Shawn
  Fact 1.6: My Favorite Martian stars Ray Walston
  Fact 1.7: My Favorite Martian is based on the 1960s television series of the same name
  Fact 1.8: Ray Walston starred in the 1960s television series My Favorite Martian
Sentence 2:  The film grossed $36.8 million domestically against a budget of $65 million.
  Fact 2.0: The film (My Favorite Martian) grossed $36.8 million domestically
  Fact 2.1: The film (My Favorite Martian) had a budget of $65 million

Retrieved items:
<item 1>
My Favorite Martian (film)
My Favorite Martian is a 1999 American science fiction comedy film starring Christopher Lloyd, Jeff Daniels, Daryl Hannah, Elizabeth Hurley, Wallace Shawn and
Ray Walston, based on the 1960s television series of the same name in which Walston starred.

 It was directed by Donald Petrie and written by Sherri Stoner and Deanna Oliver, based on the television series created by John L. Greene.

 Creatures were created by Amalgamated Dynamics from designs by Jordu Schell.

 The film grossed $36.8 million domestically against a budget of $65 million.
</item 1>

<item 2>
Remember the Titans
Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.

 The screenplay, written by Gregory Allen Howard, is based on the true story of African-American coach Herman Boone, portrayed by Denzel Washington, and his
attempt to integrate the T. C. Williams High School football team in Alexandria, Virginia, in 1971.

 Will Patton portrays Bill Yoast, Boone's assistant coach.

 Real-life athletes Gerry Bertier and Julius Campbell are portrayed by Ryan Hurst and Wood Harris, respectively.
</item 2>

<item 3>
Honey, I Blew Up the Kid
Honey, I Blew Up the Kid is a 1992 American comedy science fiction film and the sequel to the 1989 film "Honey, I Shrunk the Kids".

 Directed by Randal Kleiser and released by Walt Disney Pictures, the film stars Rick Moranis, Marcia Strassman, Robert Oliveri and Amy O'Neill, who reprise
their roles as Wayne, Diane, Nick, and Amy Szalinski respectively, as well as newcomer Keri Russell as Mandy Park, Nick's love interest and babysitter of Adam,
the Szalinskis' new two-year-old son, whose accidental exposure to Wayne's new industrial-sized growth machine causes him to gradually grow to enormous size.

 Made only three years after "Honey, I Shrunk the Kids", this film is set five years after the events depicted in the previous film.

 It was filmed in 1991.
</item 3>

<item 4>
List of accolades received by The Hurt Locker
Summit Entertainment then released "The Hurt Locker" to more screens.

 The film grossed $49.2 million worldwide, and was a success against its budget of $15 million.
</item 4>

<item 5>
McFarland, USA
McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by
Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.

 The film was co-produced by Walt Disney Pictures and Mayhem Pictures.

 Based on the true story of a 1987 cross country team from a mainly Latino high school in McFarland, California, the film stars Kevin Costner as Jim White, the
school's coach, who leads the team to win a state championship.

 The film also stars Maria Bello and Morgan Saylor.
</item 5>

<item 6>
Superman Returns
Superman Returns is a 2006 American superhero film directed and produced by Bryan Singer.

 It is based on the DC Comics character Superman and serves as an homage sequel to the motion pictures "Superman" (1978) and "Superman II" (1980), while
ignoring the events of "Superman III" (1983) and "" (1987).

 The film stars Brandon Routh as Clark Kent/Superman, Kate Bosworth as Lois Lane, Kevin Spacey as Lex Luthor, with James Marsden, Frank Langella, and Parker
Posey.

 The film tells the story of the title character returning to Earth after a five-year absence.

 He finds that his love interest Lois Lane has moved on with her life, and that his archenemy Lex Luthor is plotting a scheme that will destroy Superman and the
world.
</item 6>

<item 7>
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
</item 7>

<item 8>
Underdog (film)
Underdog is a 2007 American family superhero comedy film based on the 1960s cartoon series of the same name.

 Directed by Frederik Du Chau and written by Joe Piscatella, Adam Rifkin, and Craig A. Williams, the film stars Jim Belushi, Peter Dinklage, John Slattery, and
Patrick Warburton with the voice talents of Jason Lee, Amy Adams, and Brad Garrett.

 Unlike the TV series, the Underdog character is portrayed as a regular dog rather than an anthropomorphic one.

 Underdog, voiced by Jason Lee, was played by a lemon beagle named Leo sporting a red sweater and a blue cape.

 The film grossed $65.3 million worldwide.
</item 8>

<item 9>
The Black Hole
The Black Hole is a 1979 American science fiction film directed by Gary Nelson and produced by Walt Disney Productions.

 The film stars Maximilian Schell, Robert Forster, Joseph Bottoms, Yvette Mimieux, Anthony Perkins, and Ernest Borgnine, while the voices of the main robot
characters are provided by Roddy McDowall and Slim Pickens (both unbilled).

 The music for the film was composed by John Barry.

 It was the first film from Walt Disney Productions to receive a PG rating.

 The film was released on December 18, 1979 in the United Kingdom and on December 21, 1979 in the United States.
</item 9>

<item 10>
Titan A.E.
Titan A.E. is a 2000 American animated science fiction film directed by Don Bluth and Gary Goldman.

 Its title refers to the spacecraft central to the plot, with "A.E." meaning "After Earth".

 It stars Matt Damon, Bill Pullman, John Leguizamo, Nathan Lane, Janeane Garofalo and Drew Barrymore.

 The film's animation technique combines traditional hand-drawn animation and extensive use of computer generated imagery.

 Its working title was "Planet Ice".

 It was theatrically released on June 16, 2000, by 20th Century Fox and was the final film for Fox Animation Studios.

 The film grossed $36.8 million on a $75–$90 million budget, making a $100-million loss for the studio.
</item 10>
=== END ITEM id=f4e003f533444938 ===

=== ITEM id=f855f08477e40e6a ===
Gold sentences and facts:
Sentence 0:  The station is owned by Gray Television, as part of a duopoly with CBS affiliate KWCH-DT (channel 12); Gray also operates Univision affiliate KDCU-DT (channel 31) under a joint sales agreement with owner Entravision Communications Corporation.
  Fact 0.0: The station is owned by Gray Television.
  Fact 0.1: The station is part of a duopoly with CBS affiliate KWCH-DT (channel 12).
  Fact 0.2: Gray Television also operates Univision affiliate KDCU-DT (channel 31).
  Fact 0.3: Gray Television operates KDCU-DT under a joint sales agreement with its owner, Entravision Communications Corporation.
Sentence 1: KWCH-DT, virtual channel 12 (UHF digital channel 19), is a CBS-affiliated television station serving Wichita, Kansas, United States that is licensed to Hutchinson.
  Fact 1.0: KWCH-DT is virtual channel 12.
  Fact 1.1: KWCH-DT broadcasts on UHF digital channel 19.
  Fact 1.2: KWCH-DT is a CBS-affiliated television station.
  Fact 1.3: KWCH-DT serves Wichita, Kansas, United States.
  Fact 1.4: KWCH-DT is licensed to Hutchinson.

Retrieved items:
<item 1>
WTLV
WTLV, virtual channel 12 (VHF digital channel 13), is an NBC-affiliated television station licensed to Jacksonville, Florida, United States.

 The station is owned by the Tegna Media division of Tegna, Inc., as part of a duopoly with Orange Park-licensed ABC affiliate WJXX (channel 25).

 The two stations share studio facilities on East Adams Street (near EverBank Field) in downtown Jacksonville; WTLV's transmitter is located on Eve Drive in the
city's Killarney Shores section.
</item 1>

<item 2>
KDCU-DT
KDCU-DT, virtual and UHF digital channel 31, is a Univision-affiliated television station serving Wichita, Kansas, United States that is licensed to Derby.

 The station is owned by the Entravision Communications Corporation; Gray Television, which owns CBS affiliate KWCH-DT (channel 12) and CW affiliate KSCW-DT
(channel 33), operates KDCU under a joint sales agreement.

 KDCU's offices and master control operations are based at KWCH and KSCW's joint studio facilities on 37th Street in northeast Wichita; KDCU maintains
transmitter facilities located in rural northwestern Sedgwick County (northeast of Colwich).

 On cable, the station is available in standard definition on Cox Communications channel 15 and AT&T U-verse channel 31, and in high definition on Cox digital
channel 2015 and U-verse channel 1031.
</item 2>

<item 3>
KMBC-TV
KMBC-TV, virtual channel 9 (UHF digital channel 29), is the ABC-affiliated television station licensed to Kansas City, Missouri, United States and also serving
Kansas City, Kansas.

 The station is owned by the Hearst Television subsidiary of the Hearst Corporation, as part of a duopoly with CW affiliate KCWE (channel 29).

 The two stations share studio facilities located at the Winchester Business Center on Winchester Avenue and East 63rd Street (near Swope Park, off of I-435) in
southeastern Kansas City, Missouri; KMBC-TV maintains transmitter facilities located near the Blue River in eastern Kansas City.

 On cable, KMBC-TV is available on Charter Spectrum, Comcast Xfinity and Consolidated Communications channel 12, and Google Fiber and AT&T U-verse channel 9.

 There is a high definition feed provided on Spectrum digital channel 1200, Xfinity channel 812, Consolidated channel 610 and U-verse channel 1009.
</item 3>

<item 4>
Spectrum News Rochester and Buffalo are Charter-owned 24-hour regional news channels covering Rochester/Finger Lakes and Buffalo/Western New York; Buffalo also
supplies Jamestown’s mixed-content Cable 8 News feed. Milwaukee’s WDJT-TV is a Weigel-owned CBS station in a four-station group. Chicago’s Fox-owned WFLD is
paired with WPWR-TV. Jacksonville’s Tegna-owned NBC affiliate WTLV and ABC affiliate WJXX share studios and form a duopoly
</item 4>

<item 5>
KXMB-TV
KXMB-TV, channel 12, is the CBS-affiliated television station for Bismarck, North Dakota.

 It broadcasts a high definition digital signal on VHF channel 12 (or virtual channel 12.1 via PSIP) from a transmitter near St. Anthony.

 The station can also be seen on Midcontinent cable channel 11 in Bismarck-Mandan, and cable channel 12 in most other areas.

 There is a high definition feed provided on Midcontinent digital channel 611.
</item 5>

<item 6>
KSCW-DT
KSCW-DT, virtual channel 33 (VHF digital channel 12), is a CW-affiliated television station licensed to Wichita, Kansas, United States.

 The station is owned by Gray Television, as part of a duopoly with CBS affiliate KWCH-DT (channel 12); Gray also operates Univision affiliate KDCU-DT (channel
31) under a joint sales agreement with owner Entravision Communications Corporation.

 KSCW and KWCH share studio facilities and KDCU's master control operations are located on East 37th Street North in northeastern Wichita; KSCW maintains
transmitter facilities located in rural northeastern Reno County (due south of Buhler).
</item 6>

<item 7>
WDJT-TV
WDJT-TV, virtual channel 58 (UHF digital channel 46), is a CBS-affiliated television station licensed to Milwaukee, Wisconsin, United States.

 The station is owned by Chicago-based Weigel Broadcasting, as part of a duopoly with Racine-licensed independent station WMLW-TV (channel 49), and is also a
sister station to MeTV owned-and-operated station WBME-CD (channel 41, which is also carried via WDJT's DT2 subchannel) and Telemundo affiliate WYTU-LD (channel
63).

 All four stations share studio facilities located on South 60th Street in Milwaukee (near West Allis); WDJT's transmitter is located in Milwaukee's Lincoln
Park (next to the transmitter belonging to ABC affiliate WISN-TV, channel 12).
</item 7>

<item 8>
WJXX
WJXX, virtual channel 25 (VHF digital channel 10), is an ABC-affiliated television station serving Jacksonville, Florida, United States that is licensed to
Orange Park.

 The station is owned by the Tegna Media division of Tegna, Inc., as part of a duopoly with Jacksonville-licensed NBC affiliate WTLV (channel 12) (ironically a
former ABC affiliate itself from 1980 to 1988).

 The two stations share studio facilities on East Adams Street (near EverBank Field) in downtown Jacksonville; WJXX's transmitter is located on Eve Drive in the
city's Kilarney Shores section.
</item 8>

<item 9>
KWCH-DT
KWCH-DT, virtual channel 12 (UHF digital channel 19), is a CBS-affiliated television station serving Wichita, Kansas, United States that is licensed to
Hutchinson.

 The station is owned by Gray Television, as part of a duopoly with CW affiliate KSCW-DT (channel 33); Gray also operates Univision affiliate KDCU-DT (channel
31) under a joint sales agreement with owner Entravision Communications Corporation.

 KWCH and KSCW share studio facilities and KDCU's master control operations are located on 37th Street in northeast Wichita; KWCH maintains transmitter
facilities located in rural northeastern Reno County (south-southeast of Buhler).
</item 9>

<item 10>
KHSL-TV
KHSL-TV, virtual channel 12 (UHF digital channel 43) is a CBS-affiliated television station located in Chico, California, United States, that also serves
Redding.

 The station is owned by Heartland Media as part of a duopoly with NBC affiliate KNVN (channel 24), which is owned by Maxair Media but operated by Heartland
through a local marketing agreement.

 The two stations share studios at the corner of Eaton Road and Silverbell Road on the northwest side of Chico, while its transmitter is located along Cohasset
Road in rural Butte County northwest of Paradise.
</item 10>
=== END ITEM id=f855f08477e40e6a ===

=== ITEM id=1871edb0b77a42a5 ===
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
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
</item 1>

<item 2>
Operation Diadem
Operation Diadem, also referred to as the Fourth Battle of Monte Cassino or, in Canada, the Battle of the Liri Valley, was an offensive operation undertaken by
the Allies of World War II (U.S. Fifth Army and British Eighth Army in May 1944, as part of the Italian Campaign of World War II.

 "Diadem" was supported by air attacks called Operation Strangle.

 The opposing force was the German 10th Army.
</item 2>

<item 3>
Battle of Guam (1944)
The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands
captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
</item 3>

<item 4>
Battle of Okinawa
The Battle of Okinawa (Japanese: 沖縄戦 , Hepburn: Okinawa-sen ) (Okinawan: 沖縄戦 , "Uchinaa ikusa " ), codenamed Operation Iceberg, was a major battle of the
Pacific War fought on the island of Okinawa by United States Marine and Army forces against the Imperial Japanese Army and included the largest amphibious
assault in the Pacific War during World War II, the April 1, 1945, invasion of Okinawa itself.

 The 82-day battle lasted from April 1 until June 22, 1945.

 After a long campaign of island hopping, the Allies were planning to use Okinawa, a large island only 340 mi away from mainland Japan, as a base for air
operations for the planned invasion of the Japanese home islands.
</item 4>

<item 5>
Operation Kikusui
Operation Kikusui (菊水作戦 , Kikusui sakusen ) was a series of suicidal air attacks by Imperial Japanese forces during the Battle of Okinawa against Allied fleets
in the waters around Okinawa, as part of Operation Ten-Go.

 The name of the operation, "Kikusui", comes from the hata-jirushi of the samurai Kusunoki Masashige.
</item 5>

<item 6>
Operation Diadem order of battle
Operation Diadem order of battle is a listing of the significant formations that were involved in the fighting on the Winter Line and at the Anzio bridgehead
south of Rome during "Operation Diadem" in May - June 1944 which resulted in the Allied breakthrough at Cassino and the breakout at Anzio leading to the capture
of Rome.
</item 6>

<item 7>
Battle of Attu
The Battle of Attu, which took place from 11–30 May 1943, was a battle fought between forces of the United States, aided by Canadian reconnaissance and
fighter-bomber support, and the Empire of Japan on Attu Island off the coast of the Territory of Alaska as part of the Aleutian Islands Campaign during the
American Theater and the Pacific Theater and was the only land battle of World War II fought on incorporated territory of the United States.

 It is also the only land battle in which Japanese and American forces fought in Arctic conditions.
</item 7>

<item 8>
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
</item 8>

<item 9>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 9>

<item 10>
Operation Epsom
Operation Epsom, also known as the First Battle of the Odon, was a British Second World War offensive that took place between 26 and 30 June 1944, during the
Battle of Normandy.

 The offensive was intended to outflank and seize the German-occupied city of Caen, an important Allied objective, in the early stages of Operation Overlord,
the Allied invasion of north-west Europe.
</item 10>
=== END ITEM id=1871edb0b77a42a5 ===

=== ITEM id=2b9703a4fad076b1 ===
Gold sentences and facts:
Sentence 0: Paul Clyne was the District Attorney of Albany County, New York from January 2001 through December 2004.
  Fact 0.0: Paul Clyne was the District Attorney of Albany County, New York.
  Fact 0.1: Paul Clyne held this office from January 2001.
  Fact 0.2: Paul Clyne held this office through December 2004.
Sentence 1: P. David Soares (born October 26, 1969, Brava, Cape Verde) is the Albany County, N.Y. District Attorney.
  Fact 1.0: P. David Soares was born on October 26, 1969.
  Fact 1.1: P. David Soares was born in Brava, Cape Verde.
  Fact 1.2: P. David Soares is the District Attorney of Albany County, New York.

Retrieved items:
<item 1>
Paul Clyne
Paul Clyne was the District Attorney of Albany County, New York from January 2001 through December 2004.

 A graduate of Albany Law School, he spent about 14 years as an assistant district attorney, before he was tapped by local politicians to replace the retiring
District Attorney, Sol Greenberg.

 He was defeated for re-election by David Soares, first in the Democratic Party primary election in September 2004, and then in the general election in November
2004, in which he ran on an independent line.

 After a stint teaching at the New York Prosecutors Institute, he went into private practice as a criminal defense attorney in 2007, with an office in Albany,
New York.
</item 1>

<item 2>
Peter Sloly
Peter John M. Sloly, {'1': ", '2': ", '3': ", '4': "} (born 5 August 1966) is a former Deputy Chief of the Toronto Police Service (Divisional Policing Command
and Operational Support Command 2009-2016) and was a member of the Toronto Police for 27 years.

 He is also a former soccer player who earned 1 cap for the Canadian national side in 1985.

 He attended McMaster University, and after retiring as a player became a police officer eventually rising to the position of deputy chief on 22 September 2009.

 In 2015, he was considered a serious candidate to succeed Bill Blair as Chief of Police but was passed over in favour of Mark Saunders.
</item 2>

<item 3>
Stewart Spiers
Stewart Spiers was a small but innovative firm of plane-makers in Scotland, founded first of all in Ayr in Ayrshire and continuing under the registered name of
Stewart Speirs Ltd ["sic"] in Paisley, Renfrewshire, from c. 1933 until its demise in the mid to late 1930s.

 Like the Glasgow firm of Alexander Mathieson & Sons, Spiers benefited hugely from the thriving industries on the Firth of Clyde in the latter half of the
nineteenth century.
</item 3>

<item 4>
Skip Bolen
Skip Bolen is a Southern photographer of musicians, architecture, lifestyle and the culture of New Orleans.

 Born in Lafayette, Louisiana, he moved to New Orleans where he began his publishing career as a designer and art director.

 After moving to New York City, he began working at "House & Garden," renamed "HG," as Senior Designer in January 1988 with Anna Wintour and Alexander Liberman
at Condé Nast Publications.

 Spending evenings in jazz clubs, he began photographing jazz musicians in New York City and often when he regularly returned to New Orleans.

 After three years at Condé Nast Publications in New York City, he returned to New Orleans to pursue his jazz photography full-time.

 In 1998, he moved to Los Angeles where he became art director of House of Blues for seven years while photographing at night and weekends.

 He continued photographing jazz musicians and had his first major solo exhibition at the Jazz Bakery in Los Angeles on August 9, 2002.

 On July 4, 2006, he returned to New Orleans to pursue photography full-time documenting the recovery and rebuilding of New Orleans since Hurricane Katrina,
documenting the jazz scene, night-time photography and other photographic projects.
</item 4>

<item 5>
Doug Clifford
Douglas "Cosmo" Clifford (born April 24, 1945 in Palo Alto, California) is an American drummer, best known as a founding member of Creedence Clearwater Revival.

 After the group dissolved in the early 1970s, Clifford released a solo album and later joined CCR bassist Stu Cook in the Don Harrison Band.

 In 1995, Clifford and Cook formed the band Creedence Clearwater Revisited, performing live versions of Creedence Clearwater Revival songs.
</item 5>

<item 6>
City Cruises
City Cruises is a public limited company that operates scheduled public sightseeing cruises on the River Thames in London, serving Westminster, London Eye,
Tower and Greenwich piers every day throughout the year.

 It operates the services mainly using purpose-built wheelchair-accessible 'RiverLiner' vessels, which have a large open top deck and enclosed lower saloon with
panoramic windows and bars, backed up by some ex-Paris single-deck conservatory-style sightseeing boats.

 The company also owns a number of other boats with which it provides a variety of private charter and waterage services on the River Thames.

 City Cruises owns the London Showboat brand, under which it operates an evening dinner, cabaret and dance cruise from Westminster Pier throughout the year, and
also owns and operates the Restaurant Ship Hispaniola, which is permanently moored alongside Victoria Embankment close to Embankment Underground station in
central London.
</item 6>

<item 7>
David Soares
P. David Soares (born October 26, 1969, Brava, Cape Verde) is the Albany County, N.Y. District Attorney.

 He is a Democrat.
</item 7>

<item 8>
Scott Treleaven
Scott Treleaven is a Canadian artist whose work employs a variety of media including collage, film, video, drawing, photography and installation.
</item 8>

<item 9>
Paul Soniat
Paul Soniat is the Director of the New Orleans City Park Botanical Gardens.

 He was born at the old Touro Hospital in New Orleans.

 His family has been in New Orleans since 1727 and he grew up and later lived on the street bearing his family name.

 Paul is a self-taught piano player, and has released two CDs, absorbing the sights, sounds, and flavor of New Orleans.

 His first CD, titled "Born in New Orleans", was released in April 2005.

 His second CD was released after Hurricane Katrina hit the Gulf coast and the New Orleans' levees failed.

 This second CD is appropriately titled "Below the Water Line".
</item 9>

<item 10>
Câmara de Lobos
Câmara de Lobos (] ; literally, Portuguese: "chamber of the sealions" ) is a municipality, parish and town in the south-central coast of the island of Madeira.

 Technically a suburb of the much larger capital city of Funchal, it is one of the larger population centres and an extension of the Funchal economy.
</item 10>
=== END ITEM id=2b9703a4fad076b1 ===

=== ITEM id=1efcef6c636d0a3a ===
Gold sentences and facts:
Sentence 0: Leitneria floridana (corkwood), the sole species in the genus Leitneria, is a deciduous dioecious shrub or small tree, found only in the southeastern United States states of Arkansas, Florida, Georgia, Missouri and Texas.
  Fact 0.0: Leitneria floridana is also called corkwood.
  Fact 0.1: Leitneria floridana is the sole species in the genus Leitneria.
  Fact 0.2: Leitneria floridana is a deciduous dioecious shrub or small tree.
  Fact 0.3: Leitneria floridana is found only in the southeastern United States states of Arkansas, Florida, Georgia, Missouri and Texas.
Sentence 1: Tricyrtis is a genus of Asian flowering plants in the lily family, with approximately 20 known species.
  Fact 1.0: Tricyrtis is a genus of Asian flowering plants.
  Fact 1.1: Tricyrtis is in the lily family.
  Fact 1.2: Tricyrtis has approximately 20 known species.

Retrieved items:
<item 1>
Zygotritonia
Zygotritonia is a genus of herbaceous, perennial and bulbous plants in the Iris family (Iridaceae).

 It contains four species distributed in Tropical Africa.

 The genus name is derived from the word zygomorphic (meaning bilabiate), and the apparent resemblance to some species in the genus "Tritonia".
</item 1>

<item 2>
Tricyrtis
Tricyrtis is a genus of Asian flowering plants in the lily family, with approximately 20 known species.

 The species are commonly known in English as toad lilies.

 The genus has a native range from the Himalayas to eastern Asia, including China, Japan, Philippines and Taiwan, and a few species are cultivated for their
ornamental qualities in other parts of the world.
</item 2>

<item 3>
Tritoniopsis
Tritoniopsis is a genus of flowering plants in the family Iridaceae, first described as a genus in 1926.

 The entire genus is endemic to Cape Province in South Africa.

 The genus name refers to the African genus "Tritonia" and is combined with the Greek word "opsis", meaning "look-alike".
</item 3>

<item 4>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 4>

<item 5>
Leitneria
Leitneria floridana (corkwood), the sole species in the genus Leitneria, is a deciduous dioecious shrub or small tree, found only in the southeastern United
States states of Arkansas, Florida, Georgia, Missouri and Texas.
</item 5>

<item 6>
Ericaceae
The Ericaceae are a family of flowering plants, commonly known as the heath or heather family, found most commonly in acid and infertile growing conditions.

 The family is large, with c. 4250 known species spread across 124 genera, making it the 14th most species-rich family of flowering plants.

 The many well-known and economically important members of the Ericaceae include the cranberry, blueberry, huckleberry, rhododendron (including azaleas), and
various common heaths and heathers ("Erica", "Cassiope", "Daboecia", and "Calluna" for example).
</item 6>

<item 7>
Tritonia (plant)
Tritonia (flame freesia) is a genus of flowering plants in the iris family first described as a genus in 1802.

 They are naturally distributed across southern Africa, with a high concentration of species in Cape Province of western South Africa.

 The genus is closely related to the genus "Ixia".
</item 7>

<item 8>
Leycesteria
Leycesteria is a genus of flowering plants in the honeysuckle family Caprifoliaceae, native to temperate Asia in the Himalaya and southwestern China.
</item 8>

<item 9>
Scoliopus
Scoliopus, or fetid adderstongue, is a genus of plant within the Liliaceae family consisting of two species, "Scoliopus bigelovii" and "S. hallii."

 Both are found in deep shaded forests, primarily in the coastal counties of the western United States from central California to northern Oregon.

 The name ""Scoliopus"" derives from the Greek words "skolios" and "pous", meaning curved foot, a reference to the shape of the pedicel.

 Taxonomists believe that "Scoliopus" is closely related to "Calochortus", "Prosartes", "Streptopus" and "Tricyrtis", which all have creeping rhizomes as well
as styles that divide at the tip.
</item 9>

<item 10>
Cyrtanthus
Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
</item 10>
=== END ITEM id=1efcef6c636d0a3a ===
