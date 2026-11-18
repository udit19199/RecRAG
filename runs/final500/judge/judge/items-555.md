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

=== ITEM id=d41f5e97cf509922 ===
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
Women's football at the 2015 Pacific Games
The 2015 Pacific Games women's football tournament was the 4th edition of the Pacific Games women's football tournament.

 The women's football tournament was held in Port Moresby, Papua New Guinea between 6–16 July 2015 as part of the 2015 Pacific Games.

 The tournament was open to full women's national teams (unlike the men's tournament, which was age-restricted).
</item 1>

<item 2>
EAFF E-1 Football Championship
EAFF E-1 Football Championship, known as the East Asian Football Championship from 2003 to 2010, and the EAFF East Asian Cup for the 2013 and 2015 editions, is
a men's international football competition in East Asia for member nations of the East Asian Football Federation (EAFF).

 Before the EAFF was founded in 2002, the Dynasty Cup was held between the East Asian top four teams, and was regarded as the unofficial East Asian
Championship.

 There is a separate competition for both men (first held in 2003) and women (first held in 2005).

 There was also a combined points competition in 2005, where the results of the men's and women's teams are added together (not including qualifiers).
</item 2>

<item 3>
Australia women's national soccer team
The Australian women's national soccer team represents Australia in international women's soccer at the senior level.

 The team is overseen by the governing body for soccer in Australia, Football Federation Australia (FFA), which is currently a member of the Asian Football
Confederation (AFC) and the regional ASEAN Football Federation (AFF) since leaving the Oceania Football Confederation (OFC) in 2006.

 The team's official nickname is the Matildas (from the Australian folk song "Waltzing Matilda"), having been known as the Female Socceroos before 1995.

 The current team manager is Alen Stajcic.
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
1999 Italian Open – Women's Doubles
The 1999 Italian Open – Doubles was the doubles event of the fifty-fifth edition of the tennis tournament played at Rome, Italy, the most prestigious tennis
tournament in Southern Europe.

 It was the fifth WTA Tier I tournament of the year, and part of the European claycourt season.

 Virginia Ruano Pascual and Paola Suárez were the defending champions, but lost to Martina Hingis and Anna Kournikova in the quarterfinals.
</item 7>

<item 8>
Gambia women's national football team
The Gambia women's national football team represents the Gambia in international football competition.

 The team, however, has not competed in a match recognised by FIFA, the sport's international governing body, despite that organised women's football has been
played in the country since 1998.

 The Gambia has two youth teams, an under-17 side that has competed in FIFA U-17 Women's World Cup qualifiers, and an under-19 side that withdrew from regional
qualifiers for an under-19 World Cup.

 The development of a national team faces challenges similar to those across Africa, although the national football association has four staff members focusing
on women's football.
</item 8>

<item 9>
EAFF E-1 Football Championship (women)
EAFF E-1 Football Championship is an international football competition in East Asia for national teams of the East Asian Football Federation (EAFF).

 The competition between women's national teams is held alongside men's competition.
</item 9>

<item 10>
FIFA Women's World Cup
The FIFA Women's World Cup is an international football competition contested by the senior women's national teams of the members of "Fédération Internationale
de Football Association" (FIFA), the sport's global governing body.

 The competition has been held every four years since 1991, when the inaugural tournament, then called the Women's World Championship, was held in China.
</item 10>
=== END ITEM id=d41f5e97cf509922 ===

=== ITEM id=d67586236d582601 ===
Gold sentences and facts:
Sentence 0: Bertalan Farkas (born August 2, 1949) is the first Hungarian cosmonaut and the first Esperantist in space.
  Fact 0.0: Bertalan Farkas was born on August 2, 1949.
  Fact 0.1: Bertalan Farkas is the first Hungarian cosmonaut.
  Fact 0.2: Bertalan Farkas is the first Esperantist in space.
Sentence 1: Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
  Fact 1.0: Hans Wilhelm Schlegel was born in Überlingen on 3 August 1951.
  Fact 1.1: Hans Wilhelm Schlegel is a German physicist.
  Fact 1.2: Hans Wilhelm Schlegel is an ESA astronaut.
  Fact 1.3: Hans Wilhelm Schlegel is a veteran of two NASA Space Shuttle missions.

Retrieved items:
<item 1>
Helmuth von Moltke the Younger
Helmuth Johann Ludwig von Moltke (] ; 23 May 1848 – 18 June 1916), also known as Moltke the Younger, was a nephew of "Generalfeldmarschall" (Field Marshal)
Helmuth Karl Bernhard von Moltke and served as the Chief of the German General Staff from 1906 to 1914.

 The two are often differentiated as "Moltke the Elder" and "Moltke the Younger".
</item 1>

<item 2>
Karl Wilhelm Friedrich Schlegel
Karl Wilhelm Friedrich (after 1814: von) Schlegel (10 March 1772 – 12 January 1829), usually cited as Friedrich Schlegel, was a German poet, literary critic,
philosopher, philologist and Indologist.

 With his older brother, August Wilhelm Schlegel, he was one of the main figures of the Jena romantics.

 He was a zealous promoter of the Romantic movement and inspired Samuel Taylor Coleridge, Adam Mickiewicz and Kazimierz Brodziński.

 Schlegel was a pioneer in Indo-European studies, comparative linguistics, morphological typology, and was the first to notice what became known as Grimm's law.
</item 2>

<item 3>
Hans von Hentig
Hans von Hentig (9 June 1887 in Berlin – 6 July 1974 in Bad Tölz) was a German criminal psychologist and politician.He was the second son of lawyer Otto von
Hentig (1852–1934) .

His older brother was later diplomat Werner Otto von Hentig.

 Otto von Hentig was one of the leading lawyers in Berlin.

 Hans von Hentig was instrumental in the setting up of a short-lived Bavarian Soviet Republic in 1919.

 During the 1920s he was a prominent exponent of National Bolshevism.

 He emigrated to United States in 1935.

 Hans von Hentig worked on Yale University.
</item 3>

<item 4>
Bertalan Farkas
Bertalan Farkas (born August 2, 1949) is the first Hungarian cosmonaut and the first Esperantist in space.

 He is currently the president of Airlines Service and Trade.

 With Charles Simonyi's travel, Farkas is no longer the only Hungarian who has been to space (he is still the only astronaut, as Simonyi flew as a space
tourist).
</item 4>

<item 5>
Bernhard Bötel
Bernhard Bötel (1883–1953) was a German operatic tenor and actor who had an active career in Germany and Austria during the first half of the 20th century.

 He made recordings for several record labels during the early years of the recording industry, including His Master's Voice, Odeon Records, Polydor Records,
Tri-Ergon, and Vox Records.

 On the stage he sang a variety of roles in operas and operettas from leading parts to comprimario roles.
</item 5>

<item 6>
François Englert
François Baron Englert (] ; born 6 November 1932) is a Belgian theoretical physicist and 2013 Nobel prize laureate (shared with Peter Higgs).

 He is Professor emeritus at the Université libre de Bruxelles (ULB) where he is member of the Service de Physique Théorique.

 He is also a Sackler Professor by Special Appointment in the School of Physics and Astronomy at Tel Aviv University and a member of the Institute for Quantum
Studies at Chapman University in California.

 He was awarded the 2010 J. J. Sakurai Prize for Theoretical Particle Physics (with Gerry Guralnik, C. R. Hagen, Tom Kibble, Peter Higgs, and Robert Brout), the
Wolf Prize in Physics in 2004 (with Brout and Higgs) and the High Energy and Particle Prize of the European Physical Society (with Brout and Higgs) in 1997 for
the mechanism which unifies short and long range interactions by generating massive gauge vector bosons.

 He has made contributions in statistical physics, quantum field theory, cosmology, string theory and supergravity.
</item 6>

<item 7>
Hans Aschenborn
Hans Anton Aschenborn (1 February 1888 – 10 April 1931) was a renowned animal painter of African wildlife.

 He worked both in Germany and in southern Africa.

 His work is featured in the older German Thieme-Becker or Saur art encyclopedia.

 A Master of Arts thesis by Karin Skawran concerning the graphic works of Hans Anton Aschenborn was published in the South African art and culture periodical,
"Lantern" in 1965.)

 In 1963 the University of Pretoria published a book about Aschenborn as an artist entitled, "Hans Anton Aschenborn : Mens en Kunstenaar".

 In 1970 another one followed by the Pretoria Art Museum (South Africa).

 Other publications on Aschenborn feature his etchings and linocuts.
</item 7>

<item 8>
Richard Genée
Franz Friedrich Richard Genée (February 7, 1823 – June 15, 1895) was a Prussian born Austrian librettist, playwright, and composer.

 He is most famous for the libretto of "Die Fledermaus", Johann Strauss II's most famous operetta.

 He co-wrote the libretto without having met top-billed librettist Karl Haffner, who constructed the new story based on a play by Henri Meilhac and Ludovic
Halévy, which was considered too shocking to perform outside Paris.

 Genée, however, wrote the operetta's actual text and drew nothing from Haffner beyond the names of the characters.
</item 8>

<item 9>
Béla Bartók
Béla Viktor János Bartók ( ; ] ; 25 March 1881 – 26 September 1945) was a Hungarian composer, pianist and an ethnomusicologist.

 He is considered one of the most important composers of the 20th century; he and Liszt are regarded as Hungary's greatest composers .

 Through his collection and analytical study of folk music, he was one of the founders of comparative musicology, which later became ethnomusicology.
</item 9>

<item 10>
Hans Schlegel
Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
</item 10>
=== END ITEM id=d67586236d582601 ===

=== ITEM id=0b79cd3cc9edc847 ===
Gold sentences and facts:
Sentence 0:  The daughter of the American Consul in London, she was the first First Lady to be born outside the United States, or outside areas that were later to become part of the United States, such as the Thirteen Colonies – a distinction that would not be replicated until 192 years later by Melania Trump.
  Fact 0.0: She was the daughter of the American Consul in London.
  Fact 0.1: She was the first First Lady to be born outside the United States.
  Fact 0.2: She was the first First Lady to be born outside areas that were later to become part of the United States.
  Fact 0.3: Such areas include the Thirteen Colonies.
  Fact 0.4: The distinction of being the first First Lady born outside these areas was not replicated for 192 years.
  Fact 0.5: Melania Trump replicated this distinction 192 years later.
Sentence 1:  Before marrying future President Donald Trump in 2005, she worked as a fashion model; by 2016 she considered herself a "full-time mom."
  Fact 1.0: Before marrying future President Donald Trump in 2005, she worked as a fashion model.
  Fact 1.1: By 2016, she considered herself a 'full-time mom.'

Retrieved items:
<item 1>
Loreta Janeta Velázquez
Loreta Janeta Velázquez (June 26, 1842 – 1923), was a Cuban-born woman who masqueraded as a male Confederate soldier during the American Civil War.

 After her soldier husband's accidental death, she enlisted in the Confederate States Army in 1861.

 She then fought at Bull Run, Ball's Bluff, and Fort Donelson, but was discharged when her gender was discovered while in New Orleans.

 Undeterred, she reenlisted and fought at Shiloh, until unmasked once more.

 She then became a Confederate spy, working in both male and female guises, and as a double agent also reporting to the U.S. Secret Service.

 She remarried three more times, being widowed in each instance.

 According to William C Davis, she died in January 1923 under the name Loretta J. Beard after many years away from the public eye in a public psychiatric
facility, St. Elizabeths Hospital.

 She spied on the Union for about 5 years.
</item 1>

<item 2>
Office of the First Lady of the United States
The Office of the First Lady of the United States is the staff accountable to the First Lady of the United States.

 The office and its responsibilities, while not mandated, have grown as the role of the First Lady has grown and formalized through the history of the United
States.

 The Office of the First Lady is an entity of the White House Office, part of the Executive Office of the President.
</item 2>

<item 3>
Anna Harrison
Anna Tuthill Symmes Harrison (July 25, 1775 – February 25, 1864), wife of President William Henry Harrison and grandmother of President Benjamin Harrison, was
nominally First Lady of the United States during her husband's one-month term in 1841, but she never entered the White House.

 At the age of 65 years during her husband's presidential term, she is the oldest woman ever to become First Lady, as well as having the distinction of holding
the title for the shortest length of time, and the first person to be widowed while holding the title.

 She was the last First Lady to have been born in British America.
</item 3>

<item 4>
List of current United States first spouses
In the United States, the first spouse (first lady for women, first gentleman for men) is the term used to refer to the spouse of a chief executive - that is,
of the spouse of the President of the United States (the First Lady of the United States and the First Gentleman of the United States) and the spouses of the
governors of the 50 U.S. states and U.S. territories (Guam, Northern Mariana Islands, Puerto Rico, American Samoa, and the United States Virgin Islands) and the
spouse of the mayor of the District of Columbia.

 (The spouses of many mayors are often called the "first lady" or "first gentleman" of the city as well, and the use of the terms sometimes extends even to the
spouses of college presidents).
</item 4>

<item 5>
Vicki Risch
Vicki Risch is the former First Lady of Idaho and the wife of U.S. Sen. Jim Risch, who served as Governor of Idaho in 2006.

 She became First Lady on May 26, 2006, when her husband succeeded former Gov. Dirk Kempthorne, who resigned to become United States Secretary of the Interior.

 Mrs. Risch succeeded former First Lady Patricia Kempthorne who had held the post for over seven years.

 Mrs. Risch served as First Lady until January 2007, as her husband did not seek a full term as governor, but rather was reelected to his old post as lieutenant
governor.
</item 5>

<item 6>
List of First Ladies of the United States
The First Lady of the United States is the hostess of the White House.

 The position is traditionally filled by the wife of the President of the United States, but, on occasion, the title has been applied to women who were not
presidents’ wives, such as when the president was a bachelor or widower, or when the wife of the president was unable to fulfill the duties of the First Lady
herself.

 The First Lady is not an elected position; it carries no official duties and receives no salary.

 Nonetheless, she attends many official ceremonies and functions of state either along with or in place of the president.

 Traditionally, the First Lady does not hold outside employment while occupying the office.

 She has her own staff, including the White House Social Secretary, the Chief of Staff, the Press Secretary, the Chief Floral Designer, and the Executive Chef.

 The Office of the First Lady is also in charge of all social and ceremonial events of the White House, and is a branch of the Executive Office of the
President.
</item 6>

<item 7>
Melania Trump
Melania Trump ( ; born Melanija Knavs; ] , Germanized to Melania Knauss; April 26, 1970) is the current First Lady of the United States.

 Before marrying future President Donald Trump in 2005, she worked as a fashion model; by 2016 she considered herself a "full-time mom."
</item 7>

<item 8>
Doreen Ketchens
Doreen Ketchens (born October 3, 1966 in New Orleans, Louisiana) is a jazz clarinetist, who performs Dixieland and Trad Jazz.

 She is one of the first and few female bandleaders in New Orleans, and a musical educator.

 She has performed at concert halls and music festivals, at U.S. Embassies and decades of weekly performances in Dixieland's tradition in the Royal Street
Performing Arts Zone in the French Quarter of New Orleans with her band, Doreen's Jazz New Orleans.

 Ketchens has performed for four U.S. Presidents: Bill Clinton, George Bush Sr., Ronald Reagan and Jimmy Carter, and is widely considered one of the cultural
ambassadors of New Orleans and of the traditional music
</item 8>

<item 9>
Abigail Adams
Abigail Adams ("née" Smith; November 22  1744 – October 28, 1818) was the closest advisor and wife of John Adams, as well as the mother of John Quincy Adams.

 She is sometimes considered to have been a Founder of the United States, and is now designated as the first Second Lady and second First Lady of the United
States, although these titles were not in use at the time.
</item 9>

<item 10>
Louisa Adams
Louisa Catherine Johnson Adams (February 12, 1775 – May 15, 1852), wife of John Quincy Adams, was First Lady of the United States from 1825 to 1829.

 The daughter of the American Consul in London, she was the first First Lady to be born outside the United States, or outside areas that were later to become
part of the United States, such as the Thirteen Colonies – a distinction that would not be replicated until 192 years later by Melania Trump.
</item 10>
=== END ITEM id=0b79cd3cc9edc847 ===

=== ITEM id=511d03523c2d08aa ===
Gold sentences and facts:
Sentence 0:  His experimental works are well known and appropriately discussed in Malayalam Cinema industry and his debut movie Jalachhayam (2010) was well discussed according to its experimental approach.
  Fact 0.0: His experimental works are well known in the Malayalam Cinema industry.
  Fact 0.1: His experimental works are appropriately discussed in the Malayalam Cinema industry.
  Fact 0.2: His debut movie is Jalachhayam.
  Fact 0.3: Jalachhayam was released in 2010.
  Fact 0.4: Jalachhayam was well discussed according to its experimental approach.
Sentence 1: Jacques Feyder (] ; 21 July 1885 – 24 May 1948) was a Belgian actor, screenwriter and film director who worked principally in France, but also in the USA, Britain and Germany.
  Fact 1.0: Jacques Feyder was born on 21 July 1885.
  Fact 1.1: Jacques Feyder died on 24 May 1948.
  Fact 1.2: Jacques Feyder was a Belgian actor.
  Fact 1.3: Jacques Feyder was a Belgian screenwriter.
  Fact 1.4: Jacques Feyder was a Belgian film director.
  Fact 1.5: Jacques Feyder worked principally in France.
  Fact 1.6: Jacques Feyder also worked in the USA.
  Fact 1.7: Jacques Feyder also worked in Britain.
  Fact 1.8: Jacques Feyder also worked in Germany.

Retrieved items:
<item 1>
Sathish Kalathil
Sathish Kalathil (മലയാളം: ) is an Indian film and documentary Director and Producer in malayalam.

 He is also Story Writer, and Lyricist.

 His experimental works are well known and appropriately discussed in Malayalam Cinema industry and his debut movie Jalachhayam (2010) was well discussed
according to its experimental approach.

 Actually, his first venture was in 2008 named Veena Vaadanam, a documentary film about Art (Painting) and that was his first experimental entry in movie.

 In 2012, he directed Laloorinu Parayanullathu, a social committed documentary film about Municipal solid waste.
</item 1>

<item 2>
T. Muthuraj
T. MuthuRaj (born July 29, 1969, Coimbatore, India) is an Indian film production designer.

 He entered the film industry after completing his B.Sc.

 degree in fine arts as first assistant to Sabu Cyril after which he has served as production designer in critically acclaimed and celebrated films in major
Indian languages.

 His big break came in the 1997 Malayalam film "Guru" starring Mohanlal and Suresh Gopi, directed by Rajiv Anchal, for which he received the Kerala State Film
Award for Best Art Director.

 Muthuraj has since worked in such path breaking hits as "Arputha Theevu", "Angaditheru", "Pazhasi Raja", "Irumbukottai Murattu Singham" and "Avan Ivan".

 He has contributed to the success of 43 feature films and 185 commercials.

 He has won many film awards for best art direction and for his creative works.

 His latest work is "Nanban" (2012) directed by Shankar, which has won huge appreciation from all quarters.
</item 2>

<item 3>
Kerala School Kalolsavam
School Arts Festival of Kerala (Malayalam: കേരള സ്കൂൾ കലോത്സവം) is an annual event conducted by the state government of Kerala, featuring several art
competitions for high school and higher secondary school students of Kerala.

 The festival was started in 1956, and till 2008, it was called as "Kerala State School Youth Festival".

 The participants are students from classes 8th to 12th.

 Winners from different revenue districts for a particular event will be competing in state level competition.

 The event is usually conducted in December–January months of a year and is considered to be the biggest cultural event of Asia.

 57th edition of Kerala School Kalolsavam 2017 is scheduled for January 16 to 22 at Kannur.
</item 3>

<item 4>
Jacques Feyder
Jacques Feyder (] ; 21 July 1885 – 24 May 1948) was a Belgian actor, screenwriter and film director who worked principally in France, but also in the USA,
Britain and Germany.

 He was a leading director of silent films during the 1920s, and in the 1930s he became associated with the style of poetic realism in French cinema.

 He adopted French nationality in 1928.
</item 4>

<item 5>
Shafiq Syed
Shafiq Syed (born 1976) is a former Indian child actor turned auto rickshaw driver.

 At the age of twelve, he starred in a lead role in the Mira Nair's Academy Award-nominated film "Salaam Bombay!

" (1988), for which he also received the 1989 National Film Award for Best Child Artist at the 36th National Film Awards.

 He drives an auto rickshaw in Bangalore and also works as an assistant in television production units.
</item 5>

<item 6>
Alain Jessua
Alain Jessua (born 16 January 1932) is a French film director and screenwriter.

 He directed ten films between 1956 and 1997.

 He worked as assistant director for Jacques Becker on the set of "Casque d'or", with Max Ophüls for "Madame de..." and "Lola Montès" and with Marcel Carné on
"Wasteland". "

Léon la lune" his first short film won the influential Prix Jean Vigo in 1957.

 He directed first feature film in 1963 "La vie à l'envers" that won Best First Film at Venice Film Festival, in 1964.
</item 6>

<item 7>
Shajoon Kariyal
He debuted as a director with "Rajaputhran" (1996), starring Suresh Gopi, Shobhana and Vikram.

 He has directed many films, including the commercially successful "Thachiledathu Chundan" (1999) and the critically acclaimed "Vadakkumnadhan" (2006).

 After "Vadakkumnadhan", he planned two films, "Raman Police" and "Talkies", but both the projects did not work out.

 In 2012, he directed "Chettayees" which he also co-produced, as one of the five partners of the newly launched production house Thakkaali Films.

 His latest film is "Sir C. P." (2015).
</item 7>

<item 8>
K. M. Binu
Kalayathumkuzhi Mathews Binu (born 20 December 1980) is an Indian track and field athlete from Kerala who specializes in 400 metres and 800 metres.

 He holds the current 400 metres national record of 45.48 s set at the 2004 Athens Olympics on 20 August 2004.

 He broke the 44-year-old Olympics mark (by an Indian) held by Milkha Singh who set an Indian National Record with a timing of 45.73 s at the 1960 Rome
Olympics.

 He and his elder sister K. M. Beenamol made history when they became the first Indian siblings to win medals in a major international competition.

 They won medals at the Busan Asian Games (2002).

 While Binu won the men's 800 metres silver, his sister won the gold medal in the women's event.

 Binu received the Arjuna Award for the year 2006 for his achievements in the Indian athletics.
</item 8>

<item 9>
Jalachhayam
Jalachhayam (മലയാളം: ) is a 2010 Indian Malayalam-language Experimental film, produced and directed by Sathish Kalathil under the banner of 'The People's
Films'.

 The story was written by Sujith Aalungal and describes the warm relationship between a village man and an artist (painter) from the city.
</item 9>

<item 10>
Bandi Sailu
Bandi Sailu (1928 – 15 July 2000) or Sayilu Saayilu was a Marxist revolutionary born in a Dalit family in Mandapelli village, Duggondi mandal, Narsampeta
Thalukh in the Warangal district of Andhra Pradesh, India.
</item 10>
=== END ITEM id=511d03523c2d08aa ===

=== ITEM id=ec1bda06bffecc9c ===
Gold sentences and facts:
Sentence 0:  It is the third installment of the "Naked" series "Naked Killer" (1992) and "Naked Weapon" (2002).
  Fact 0.0: The work is the third installment of the 'Naked' series.
  Fact 0.1: 'Naked Killer' is a work in the 'Naked' series released in 1992.
  Fact 0.2: 'Naked Weapon' is a work in the 'Naked' series released in 2002.
Sentence 1: Naked Weapon (赤裸特工) is a 2002 Hong Kong action-thriller film directed by action choreographer Tony Ching and starring Maggie Q, Anya Wu and Daniel Wu.
  Fact 1.0: Naked Weapon is also titled 赤裸特工.
  Fact 1.1: Naked Weapon is a 2002 film.
  Fact 1.2: Naked Weapon is a Hong Kong film.
  Fact 1.3: Naked Weapon is an action-thriller film.
  Fact 1.4: Naked Weapon was directed by action choreographer Tony Ching.
  Fact 1.5: Naked Weapon stars Maggie Q.
  Fact 1.6: Naked Weapon stars Anya Wu.
  Fact 1.7: Naked Weapon stars Daniel Wu.

Retrieved items:
<item 1>
The Naked Gun 2½: The Smell of Fear
The Naked Gun 2½: The Smell of Fear is a 1991 comedy film.

 It is the sequel to the 1988 film "" and the second installment in "The Naked Gun" film series.

 The film stars Leslie Nielsen as the comically bumbling Police Lt. Frank Drebin of "Police Squad!

".

 Priscilla Presley plays the role of Jane, with O. J. Simpson as Nordberg and George Kennedy as police captain Ed Hocken.

 The film also features Robert Goulet (who previously made a "special guest star" appearance on "Police Squad!")

 as the villainous Quentin Hapsburg and Richard Griffiths as renewable fuel advocate Dr. Albert S. Meinheimer (as well as his evil double, Earl Hacker).

 Zsa Zsa Gabor, Mel Tormé and members of the Chicago Bears have cameo roles.
</item 1>

<item 2>
Naked Weapon
Naked Weapon (赤裸特工) is a 2002 Hong Kong action-thriller film directed by action choreographer Tony Ching and starring Maggie Q, Anya Wu and Daniel Wu.
</item 2>

<item 3>
Naked City (TV series)
Naked City is a police drama series from Screen Gems which was broadcast from 1958 to 1959 and from 1960 to 1963 on the ABC television network.

 It was inspired by the 1948 motion picture "The Naked City" and mimics its dramatic "semi-documentary" format.

 As in the film, each episode concluded with a narrator intoning the iconic line: "There are eight million stories in the naked city.

 This has been one of them."
</item 3>

<item 4>
Naked Soldier
Naked Soldier (絕色武器) is a 2012 Hong Kong action film directed by Marco Mak and starring Jennifer Tse.

 It is the third installment of the "Naked" series "Naked Killer" (1992) and "Naked Weapon" (2002).
</item 4>

<item 5>
The Human Centipede 2 (Full Sequence)
The Human Centipede 2 (Full Sequence) is a 2011 Dutch black and white exploitation-horror film written, directed, and co-produced by Dutch filmmaker Tom Six.

 The sequel to Six's 2009 film "The Human Centipede (First Sequence)", the film stars Laurence R. Harvey as a mentally impaired English man who watches and
becomes obsessed with the first "Human Centipede" film, and decides to make his own "centipede" consisting of twelve people, including Ashlynn Yennie, an
actress from the first film.
</item 5>

<item 6>
American Pie Presents: The Naked Mile
American Pie Presents: The Naked Mile (also known as American Pie: The Naked Mile) is a 2006 American sex comedy film released by Universal Pictures.

 It is the second installment in the "American Pie Presents" series and the fifth installment in the "American Pie" franchise.

 The film begins a story arc that concludes with "" (2007).

 John White stars as Erik Stifler, a high school senior who is given a "guilt free pass" by his girlfriend, Tracy Sterling (Jessy Schram), and so visits the
Beta House fraternity led by his cousin, Dwight Stifler (Steve Talley), to run a mile naked.

 Christopher McDonald co-stars as Erik's father, Harry, and Eugene Levy once again plays Jim's Dad, who turns out to be a family friend of both Erik's and
Tracy's.

 Also, it is in this film that his name is revealed to be "Noah Levenstein".
</item 6>

<item 7>
The Conman
The Conman is a 1998 Hong Kong action comedy film directed by Wong Jing and starring Andy Lau, Athena Chu and Nick Cheung.

 Despite the Chinese title, which translates as "Knight of Gamblers 1999", Andy Lau does not reprise his role as the "Knight of Gamblers" from the "God of
Gamblers" series, which was also directed by Wong Jing.

 The film was followed by a sequel "The Conmen in Vegas", which Lau and Cheung return with new cast members Natalis Chan, Kelly Lin, Meggie Yu and Alex Man.
</item 7>

<item 8>
Die Hard with a Vengeance
Die Hard with a Vengeance is a 1995 American action film and the third in the "Die Hard" film series.

 It was co-produced and directed by John McTiernan (who directed "Die Hard"), written by Jonathan Hensleigh, and stars Bruce Willis as New York City Police
Department Lieutenant John McClane, Samuel L. Jackson as McClane's reluctant partner Zeus Carver, and Jeremy Irons as Simon Gruber.

 It was released on May 19, 1995, five years after "Die Hard 2", becoming the highest-grossing film at the worldwide box-office that year, but received mixed
reviews.

 It was followed by "Live Free or Die Hard" and "A Good Day to Die Hard" in 2007 and 2013, respectively.
</item 8>

<item 9>
Mel Gibson filmography
Five years later he played Martin Riggs in the buddy cop action comedy "Lethal Weapon" alongside Danny Glovera role he later reprised in its sequels "Lethal
Weapon 2" (1989), "Lethal Weapon 3" (1992), and "Lethal Weapon 4" (1998).
</item 9>

<item 10>
Naked Gun 33⅓: The Final Insult
Naked Gun 33⅓: The Final Insult is a 1994 comedy film, and the third and final installment in "The Naked Gun" film series, which was based on the "Police Squad!

" television series.
</item 10>
=== END ITEM id=ec1bda06bffecc9c ===

=== ITEM id=1a4ba8e5b085aa9e ===
Gold sentences and facts:
Sentence 0: A Summer in the Cage is a 2007 documentary film about a man's experiences with bipolar disorder.
  Fact 0.0: A Summer in the Cage is a documentary film.
  Fact 0.1: A Summer in the Cage was released in 2007.
  Fact 0.2: A Summer in the Cage is about a man's experiences with bipolar disorder.
Sentence 1: American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.
  Fact 1.0: American Hardcore: The History of American Punk Rock 1980-1986 is a documentary.
  Fact 1.1: American Hardcore: The History of American Punk Rock 1980-1986 was directed by Paul Rachman.
  Fact 1.2: American Hardcore: The History of American Punk Rock 1980-1986 was written by Steven Blush.

Retrieved items:
<item 1>
Graph query row:
film: American Hardcore
kind: documentary feature film
relationship: CREATED_BY
connected_entity: Paul Rachman
</item 1>

<item 2>
Deadfall (1993 film)
Deadfall is a 1993 crime drama film directed by Christopher Coppola.

 Coppola co-wrote the script with Nick Vallelonga.

 The film stars Michael Biehn, Nicolas Cage, Charlie Sheen, James Coburn, and Peter Fonda.

 It is also the prime influence on the song 'Deadfall', written by the American hardcore punk band Snot.

 A prequel/sequel, "Arsenal", starring Nicolas Cage as his character Eddie King, was released in 2017.
</item 2>

<item 3>
Combat Zone Wrestling
Combat Zone Wrestling (CZW) is an American independent wrestling promotion.

 In 1999, John Zandig and five of his students, Ric Blade, T.C.K, Lobo, Nick Gage, and Justice Pain (along with trainer Jon Dahmer) began to run professional
wrestling shows in New Jersey and Delaware, showcasing a brand of hardcore wrestling dubbed as "ultraviolence".

 Ladders, tables, steel folding chairs, thumbtacks, barbed wire, weed whackers, light tubes, panes of glass, and fire are all common elements of "ultraviolent
wrestling" in CZW.

 The company filled a niche for hardcore wrestling fans that had been left open by the folding of Extreme Championship Wrestling (ECW).

 CZW established themselves as the leading American hardcore wrestling promotion at the ECW Arena with their "Cage of Death 3" show in 2001, the year ECW
folded.
</item 3>

<item 4>
Hellfest (American music festival)
Hellfest was an American hardcore punk festival.

 Named for the intense summer heat at the show, the festival originally began featuring hardcore bands before crossing into several genres later on.

 Hellfest was known for its fan-friendly atmosphere, (including a lack of barriers, providing easy access to the stage for stage diving) and for its high
social-awareness, focusing on such causes as animal rights, with many attendees following the straight edge and/or vegan lifestyle.

 It continued as an annual event until it was cancelled in 2005.

 After its cancellation less than thirty-six hours before doors were scheduled to open on the festival's ninth year, some fans, unhappy with the inadequate
processing of refunds, filed small claims court lawsuits against promoter Keith Allen.
</item 4>

<item 5>
American Pie Presents: Band Camp
American Pie Presents: Band Camp (also known as American Pie: Band Camp) is a 2005 direct-to-DVD sex comedy film.

 It is the first installment in the "American Pie Presents" series and the fourth installment in the "American Pie" franchise.

 It was directed by Steve Rash and written by Brad Riddell.

 The film centers on Matt Stifler, younger brother of Steve Stifler.

 In this film, Matt is sent to the infamous band camp, where he is compelled to change his ways.
</item 5>

<item 6>
Hardcore (1979 film)
Hardcore is a 1979 American crime drama film written and directed by Paul Schrader and starring George C. Scott, Peter Boyle and Season Hubley.

 The story concerns a father searching for his daughter, who has vanished only to appear in a pornographic film.

 Writer-director Schrader had previously written the screenplay for Martin Scorsese's "Taxi Driver", and both films share a theme of exploring an unseen
subculture.
</item 6>

<item 7>
A Summer in the Cage
A Summer in the Cage is a 2007 documentary film about a man's experiences with bipolar disorder.

 The film follows the filmmaker's friend Sam and features an interview with mental health scholar Kay Redfield Jamison.

 It was directed by Benjamin Selkow.

 The documentary debuted on the Sundance Channel in 2007.
</item 7>

<item 8>
Punk's Not Dead (2007 film)
Punk's Not Dead is a 2007 documentary film directed by Susan Dynner, an American hardcore punk fan.

 The film claims to infiltrate American clubs, malls, recording studios, etc. where it sets out to claim hardcore punk and pop punk music is "thriving" from an
American perspective.

 Its content features performances largely from 1980s hardcore bands and MTV skate punk and pop punk/rock acts.

 It also includes various interviews and behind-the-scenes footage with the bands, labels and fans.
</item 8>

<item 9>
All Ages: The Boston Hardcore Film
All Ages: The Boston Hardcore Film, a Gallery East Production in association with Stone Films/NYC, is a documentary directed by Drew Stone, which had its world
premier at the Independent Film Festival of Boston on April 27, 2012.

 The film features interviews, archival footage and the music of Boston’s early hardcore bands including Deep Wound, DYS, Gang Green, Impact Unit, Jerry's Kids,
Negative FX, SS Decontrol, The Freeze and The F.U.'s.

 Also featured in the film are interviews with renowned author Michael Patrick MacDonald ("Easter Rising", "All Souls"), Actress Christine Elise McCarthy,
Advertising Executive Jonathan Anastas, Thrasher Magazine Editor Jake Phelps, "American Hardcore" Director Paul Rachman, and Newbury Comics owner Michael
Dreese.

 "All Ages" has been released on DVD with numerous extras and director's commentary.
</item 9>

<item 10>
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 10>
=== END ITEM id=1a4ba8e5b085aa9e ===

=== ITEM id=04957e2e6c4ef02c ===
Gold sentences and facts:
Sentence 0:  One nunatak was first roughly mapped and named "Moltke Nunatak" by the Second German Antarctic Expedition of 1911–12 under Wilhelm Filchner.
  Fact 0.0: A nunatak was first roughly mapped by the Second German Antarctic Expedition of 1911–12
  Fact 0.1: The Second German Antarctic Expedition of 1911–12 was under Wilhelm Filchner
  Fact 0.2: The Second German Antarctic Expedition named the nunatak Moltke Nunatak
Sentence 1:  He named it for General Helmuth von Moltke, Chief of the German General Staff and Secretary of State for Home Affairs.
  Fact 1.0: The nunatak was named for General Helmuth von Moltke
  Fact 1.1: General Helmuth von Moltke was Chief of the German General Staff
  Fact 1.2: General Helmuth von Moltke was Secretary of State for Home Affairs
Sentence 2: Helmuth Johann Ludwig von Moltke (] ; 23 May 1848 – 18 June 1916), also known as Moltke the Younger, was a nephew of "Generalfeldmarschall" (Field Marshal) Helmuth Karl Bernhard von Moltke and served as the Chief of the German General Staff from 1906 to 1914.
  Fact 2.0: Helmuth Johann Ludwig von Moltke was born 23 May 1848
  Fact 2.1: Helmuth Johann Ludwig von Moltke died 18 June 1916
  Fact 2.2: Helmuth Johann Ludwig von Moltke is also known as Moltke the Younger
  Fact 2.3: Helmuth Johann Ludwig von Moltke was a nephew of Generalfeldmarschall (Field Marshal) Helmuth Karl Bernhard von Moltke
  Fact 2.4: Helmuth Johann Ludwig von Moltke served as Chief of the German General Staff from 1906 to 1914

Retrieved items:
<item 1>
Franz Halder
Franz Halder (30 June 1884 – 2 April 1972) was a German general and the chief of the "Oberkommando des Heeres" staff (OKH, Army High Command) from 1938 until
September 1942, when he was dismissed after frequent disagreements with Adolf Hitler.

 Until December 1941 Halder's military position corresponded to the old Chief of the General Staff position, which during World War I had been the highest
military office in the German Imperial Army.

 Halder's diary during his time as chief of OKH General Staff has been a source for authors that have written about such subjects as Hitler, World War II and
the Nazi Party.

 In William Shirer's "The Rise and Fall of the Third Reich", Halder's diary is cited hundreds of times.
</item 1>

<item 2>
Helmuth von Moltke the Younger
Helmuth Johann Ludwig von Moltke (] ; 23 May 1848 – 18 June 1916), also known as Moltke the Younger, was a nephew of "Generalfeldmarschall" (Field Marshal)
Helmuth Karl Bernhard von Moltke and served as the Chief of the German General Staff from 1906 to 1914.

 The two are often differentiated as "Moltke the Elder" and "Moltke the Younger".
</item 2>

<item 3>
Ludwig Beck
Ludwig August Theodor Beck (29 June 1880 – 21 July 1944) was a German general and Chief of the German General Staff during the early years of the Nazi regime in
Germany before World War II.

 Ludwig Beck was never a member of the Nazi Party, though in the early 1930s he supported Adolf Hitler's forceful denunciation of the Versailles Treaty and
belief in the need for Germany to rearm.

 Beck had grave misgivings regarding the Nazi demand that all German officers swear an oath of fealty to the person of Hitler in 1934, though he believed that
Germany needed strong government and that Hitler could successfully provide this so long as he was influenced by traditional elements within the military rather
than the SA and SS.
</item 3>

<item 4>
List of Generalobersts
Generaloberst, in English Colonel General, was, in Germany and Austria-Hungary—the German "Reichswehr" and "Wehrmacht", the Austro-Hungarian Common Army, and
the East German National People's Army, as well as the respective police services—the second highest general officer rank, ranking above full general but below
general field marshal.

 It was equivalent to "Generaladmiral" in the "Kriegsmarine" until 1945, or to Flottenadmiral in the "Volksmarine" until 1990.

 The rank was the highest ordinary military rank and the highest military rank awarded in peacetime; the higher rank of general field marshal was only awarded
in wartime by the head of state.

 In general, a Generaloberst had the same privileges as a general field marshal.
</item 4>

<item 5>
Ferdinand Jodl
Ferdinand Alfred Friedrich Jodl (28 November 1896 – 9 June 1956) was a German general during World War II who commanded the Mountain Corps Norway during the
Petsamo–Kirkenes Offensive.

 He was the brother of Alfred Jodl, Chief of the Operations Staff of the OKW.

 Ferdinand Jodl was acquitted by a German court.
</item 5>

<item 6>
Alfred Jodl
Alfred Josef Ferdinand Jodl (   ; 10 May 1890 – 16 October 1946) was a German general and war criminal during World War II, who served as the Chief of the
Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).
</item 6>

<item 7>
German General Staff
The German General Staff, originally the Prussian General Staff and officially Great General Staff ("Großer Generalstab"), was a full-time body at the head of
the Prussian Army and later, the German Army, responsible for the continuous study of all aspects of war, and for drawing up and reviewing plans for
mobilization or campaign.

 It existed unofficially from 1806, and was formally established by law in 1814, the first general staff in existence.

 It was distinguished by the formal selection of its officers by intelligence and proven merit rather than patronage or wealth, and by the exhaustive and
rigorously structured training which its staff officers undertook.

 Its rise and development gave the German armed forces a decisive strategic advantage over their adversaries for nearly a century and a half.
</item 7>

<item 8>
Herbert Büchs
Herbert Büchs (20 November 1913 – 19 May 1996) was a Lieutenant General of the German Air Force and a former Luftwaffe staff officer in Nazi Germany's
Oberkommando der Wehrmacht (OKW) during World War II.

 As second adjutant to General Alfred Jodl with the rank of Major he is notable for being present in the conference room when the 20 July plot bomb exploded in
1944.
</item 8>

<item 9>
Walter Dornberger
Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was a German Army artillery officer whose career spanned World War I and World War
II.

 He was a leader of Nazi Germany's V-2 rocket program and other projects at the Peenemünde Army Research Center.
</item 9>

<item 10>
Moltke Nunataks
The Moltke Nunataks ( ) are a chain of north–south trending nunataks close to the northeastern end of the Filchner Ice Shelf, Antarctica.

 One nunatak was first roughly mapped and named "Moltke Nunatak" by the Second German Antarctic Expedition of 1911–12 under Wilhelm Filchner.

 He named it for General Helmuth von Moltke, Chief of the German General Staff and Secretary of State for Home Affairs.

 Surveys during the mid-1950s by British, Argentine and United States expeditions indicate that a group of four or five nunataks exist in the area.
</item 10>
=== END ITEM id=04957e2e6c4ef02c ===

=== ITEM id=4c2590feb62d9b7f ===
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
Spengler Cup
The Spengler Cup is an annual invitational ice hockey tournament held in Davos, Switzerland.

 First held in 1923, the Spengler Cup is often cited as the oldest invitational ice hockey tournament in the world.

 The event is hosted by the Swiss team HC Davos and played each year in Davos, Switzerland, between Christmas (December 25) and New Year's Day.

 Currently, all games are held at Vaillant Arena.
</item 3>

<item 4>
2015 Spengler Cup
The 89th Spengler Cup was an ice hockey competition held in Davos, Switzerland between 26 and 31 December 2015.

 All matches were played at HC Davos's home known as Vaillant Arena.

 Six competing teams were split into two groups of three (in the round-robin series).

 The two groups, named Torriani and Cattini, are named after legendary Swiss hockey players Richard 'Bibi' Torriani and the Cattini brothers, Hans and
Ferdinand.
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
1982 Donnay Indoor Championships
The 1982 Donnay Indoor Championships was a men's tennis tournament played on indoor carpet courts in Brussels in Belgium the event was part of the 1982 Volvo
Grand Prix.

 The tournament was held from 8 March through 14 March 1982.

 Third-seeded Vitas Gerulaitis won the singles title.
</item 9>

<item 10>
Fife Flyers
Fife Flyers Ice Hockey Club is the oldest professional ice hockey club in the UK, established in 1938.

 The Flyers play their home games at Fife Ice Arena in Kirkcaldy which has a capacity of just Over 3000 (seated and standing).

 The arena is home to not only Fife Flyers, who play in the Elite Ice Hockey League but also to Kirkcaldy Junior Ice Hockey Club.

 Traditionally many of the Flyers' players have come up through the junior ranks to play at a professional level.
</item 10>
=== END ITEM id=4c2590feb62d9b7f ===

=== ITEM id=b94e0a301c60cffa ===
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
</item 1>

<item 2>
List of Walt Disney Pictures films
This is a list of films released theatrically under the Walt Disney Pictures banner (known as that since 1983, with "Never Cry Wolf" as its first release) and
films released before that under the former name of the parent company, Walt Disney Productions (1929–1983).

 Most films listed here were distributed in the United States by the company's distribution division, Walt Disney Studios Motion Pictures (formerly known as
Buena Vista Distribution Company [1953–1987] and Buena Vista Pictures Distribution [1987–2007]).

 The Disney features produced before "Peter Pan" (1953) were originally distributed by RKO Radio Pictures, and are now distributed by Walt Disney Studios Motion
Pictures.
</item 2>

<item 3>
Pinocchio (1940 film)
Pinocchio is a 1940 American animated musical fantasy film produced by Walt Disney Productions and based on the Italian children's novel "The Adventures of
Pinocchio" by Carlo Collodi.

 It was the second animated feature film produced by Disney, made after the success of "Snow White and the Seven Dwarfs" (1937).
</item 3>

<item 4>
Toy Story
Toy Story is a 1995 American computer-animated buddy comedy adventure film produced by Pixar Animation Studios for Walt Disney Pictures.

 The directorial debut of John Lasseter, "Toy Story" was the first feature-length computer-animated film and the first feature film produced by Pixar.

 Taking place in a world where anthropomorphic toys pretend to be lifeless whenever humans are present, the film's plot focuses on the relationship between
Woody, an old-fashioned pullstring cowboy doll (voiced by Tom Hanks), and Buzz Lightyear, an astronaut action figure (voiced by Tim Allen), as they evolve from
rivals competing for the affections of Andy, their owner, to friends who work together to be reunited with Andy as his family prepares to move to a new home.

 The screenplay was written by Joss Whedon, Andrew Stanton, Joel Cohen and Alec Sokolow, based on a story by Lasseter, Pete Docter, Stanton and Joe Ranft.

 The film features music by Randy Newman, and was executive-produced by Steve Jobs and Edwin Catmull.
</item 4>

<item 5>
Walt Disney's Treasury of Classic Tales
Walt Disney's Treasury of Classic Tales is an American Sunday comic strip, which ran in newspapers from 13 July 1952 until 15 February 1987.

 Each story adapted a different Disney film, such as Darby O'Gill and the Little People, Peter Pan, or Davy Crockett.

 It was run in relatively few papers with 58 in 1957 and 55 in 1966 and was principally a vehicle for promoting new and re-released Disney films.
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
Darby O'Gill and the Little People
Darby O'Gill and the Little People is a 1959 Walt Disney Productions feature film starring Albert Sharpe, Janet Munro, Sean Connery and Jimmy O'Dea, in a tale
about a wily Irishman and his battle of wits with leprechauns.

 The film was directed by Robert Stevenson and its screenplay written by Lawrence Edward Watkin after the books of Herminie Templeton Kavanagh.
</item 7>

<item 8>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
</item 8>

<item 9>
Frankie Darro
Frankie Darro (born Frank Johnson, Jr.

 December 22, 1917 – December 25, 1976) was an American actor and later in his career a stuntman.

 He began his career as a child actor in silent films, progressed to lead roles and co-starring roles in adventure, western, dramatic, and comedy films, and
later became a character actor and voice-over artist.

 He is known for his role as Lampwick, the unlucky boy turned into a donkey in Walt Disney's 2nd animated feature, "Pinocchio", which was originally released in
February 1940.
</item 9>

<item 10>
Walt Disney Pictures
Walt Disney Pictures, Inc. is an American film production company and a subsidiary of Walt Disney Studios, owned by The Walt Disney Company.

 The division is based at the Walt Disney Studios in Burbank, California, and is the main producer of live-action feature films within the Walt Disney Studios
unit.

 It took on its current name in 1983.

 Today, in conjunction with the other units of Walt Disney Studios, Walt Disney Pictures is classified as one of Hollywood's "Big Six" film studios.

 Films produced by Walt Disney Animation Studios and Pixar Animation Studios are also released under this brand.
</item 10>
=== END ITEM id=b94e0a301c60cffa ===

=== ITEM id=6946316904cf63a1 ===
Gold sentences and facts:
Sentence 0:  He earned 1 cap under Jack Charlton for the Irish national team.
  Fact 0.0: The player earned 1 cap for the Irish national team.
  Fact 0.1: The player earned that cap under Jack Charlton.
Sentence 1:  He was part of the England team that won the 1966 World Cup.
  Fact 1.0: The player was part of the England team that won the 1966 World Cup.
  Fact 1.1: The England team won the 1966 World Cup.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Arthur Milton
Clement Arthur Milton (10 March 1928 – 25 April 2007) was an English cricketer and footballer.

 He played County cricket for Gloucestershire from 1948 to 1974, playing six Test matches for England in 1958 and 1959.

 He also played domestic football for Arsenal between 1951 and 1955, and then for a brief period for Bristol City.

 He played one match for England in 1951, against Austria at Wembley.

 He was the last man, and the last survivor, of the twelve people to have played at the highest international level for both England's football and cricket
teams.
</item 2>

<item 3>
William McCourty
William McCourty (1884 – 10 December 1917) was an English professional footballer born in Morpeth, Northumberland, who played in the Football League for
Birmingham.

 McCourty, a former coal miner who played at left half, joined Birmingham in May 1909.

 He made his debut in the Second Division on 13 September 1909, deputising for the injured Tommy Daykin in a home game against Glossop which finished as a 2–2
draw, and was never picked again, returning to non-league football the following year.
</item 3>

<item 4>
Barry Murphy (footballer, born 1959)
Barry Murphy (born 1 April 1959 in Dublin) was an Irish soccer player during the 1980s.

 He earned 1 cap under Jack Charlton for the Irish national team.
</item 4>

<item 5>
Gordon Burns: Scottish ex-player and Troon manager; played for Ayr. Jack Charlton: England’s 1966 World Cup-winning defender, Leeds legend and record
appearance-holder; Bobby’s brother. Barry Murphy: Irish international, one cap under Charlton. Adnan Januzaj: Belgian winger at Real Sociedad; former Manchester
United player and 2014 World Cup participant. Arthur Milton: England footballer and cricketer, with six Test caps. Leatherbarrow, McC
</item 5>

<item 6>
Billy Morgan (footballer, born 1891)
William Albert L. Morgan (3 November 1891 – after 1927) was an English professional footballer born in Old Hill, Cradley Heath, Staffordshire, who played either
at outside left or inside left.

 He played for Birmingham, Coventry City and Crystal Palace in the Football League, and was capped once for a Football League representative side.
</item 6>

<item 7>
Canadian athletes Ross MacKenzie and Brian MacLaren earned 1966 Commonwealth and 1967 Pan American relay silver; both competed in Olympic 400m. Guyanese Harry
Prowell set a marathon record and remains Guyana’s sole Olympic marathoner. Canadian miler Dave Bailey broke four minutes and won Pan Am bronze; Siegmar
Ohlemann won Pan Am 800m silver. Geri Donnelly and China’s Liu Ailing were World Cup footballers; Liu also
</item 7>

<item 8>
Jack Charlton
John Charlton, OBE, DL (born 8 May 1935) is an English former footballer and manager who played as a defender.

 He was part of the England team that won the 1966 World Cup.

 He is the elder brother of former Manchester United forward Bobby Charlton, who was also a teammate in England's World Cup final victory.

 He spent his entire club career with Leeds United from 1950 to 1973, helping the club to the Second Division title (1963–64), First Division title (1968–69),
FA Cup (1972), League Cup (1968), Charity Shield (1969), Inter-Cities Fairs Cup (1968 and 1971), as well as one other promotion from the Second Division
(1955–56) and five second-place finishes in the First Division, two FA Cup final defeats and one Inter-Cities Fairs Cup final defeat.

 His 629 league and 762 total competitive appearances are club records.

 In 2006, Leeds United supporters voted Charlton into the club's greatest ever XI.
</item 8>

<item 9>
Clare Taylor
Clare Elizabeth Taylor MBE (born 22 May 1965) is an English sportswoman, the first woman to have played on a World Cup team in both cricket and football.

 She represented England at both cricket, as a member of the winning World Cup cricket team in 1993, and football (World Cup 1995).

 She was awarded the MBE in the Queen's Birthday Honours in 2000 for her services to women’s sport.

 Taylor attended Moor End High School and still has her name on the athletics records board.
</item 9>

<item 10>
Walter Ward (footballer)
Walter Ward (1869 – after 1890) was an English professional footballer born in Birmingham who played in the Football Alliance for Small Heath.

 During the 1890–91 season, Ward deputised for regular goalkeeper Chris Charsley when Charsley's duties as a serving police officer demanded his absence from
the side.
</item 10>
=== END ITEM id=6946316904cf63a1 ===

=== ITEM id=7070eda2cf3e88a3 ===
Gold sentences and facts:
Sentence 0:  It is the fourth and final film in the original "Superman" film series, and the first film in that series not to be produced by Alexander and Ilya Salkind, but rather by Golan-Globus' Cannon Films, in association with Warner Bros. Gene Hackman returned as Lex Luthor, who creates an evil solar-powered version of Superman called Nuclear Man.
  Fact 0.0: The film is the fourth and final film in the original Superman film series.
  Fact 0.1: The film is the first film in the original Superman series not produced by Alexander and Ilya Salkind.
  Fact 0.2: The film was produced by Golan-Globus' Cannon Films.
  Fact 0.3: The film was produced in association with Warner Bros.
  Fact 0.4: Gene Hackman returned as Lex Luthor in the film.
  Fact 0.5: Lex Luthor creates an evil solar-powered version of Superman called Nuclear Man.
Sentence 1: Eugene Allen "Gene" Hackman (born January 30, 1930) is a retired American actor and novelist.
  Fact 1.0: Eugene Allen 'Gene' Hackman was born on January 30, 1930.
  Fact 1.1: Gene Hackman is a retired American actor.
  Fact 1.2: Gene Hackman is a retired American novelist.

Retrieved items:
<item 1>
Superman IV: The Quest for Peace
Superman IV: The Quest for Peace is a 1987 American superhero film directed by Sidney J. Furie, based on the DC Comics character Superman.

 It is the fourth and final film in the original "Superman" film series, and the first film in that series not to be produced by Alexander and Ilya Salkind, but
rather by Golan-Globus' Cannon Films, in association with Warner Bros. Gene Hackman returned as Lex Luthor, who creates an evil solar-powered version of
Superman called Nuclear Man.
</item 1>

<item 2>
Charles A. Agemian
Charles A. Agemian (July 22, 1909 – March 30, 1996) was an American banker who took early retirement from Chase Manhattan Bank, where he was executive vice
president of operations, to become chairman and chief executive officer of the Hackensack Trust Co., which later was renamed Garden State National Bank.
</item 2>

<item 3>
Superman Returns
Superman Returns is a 2006 American superhero film directed and produced by Bryan Singer.

 It is based on the DC Comics character Superman and serves as an homage sequel to the motion pictures "Superman" (1978) and "Superman II" (1980), while
ignoring the events of "Superman III" (1983) and "" (1987).

 The film stars Brandon Routh as Clark Kent/Superman, Kate Bosworth as Lois Lane, Kevin Spacey as Lex Luthor, with James Marsden, Frank Langella, and Parker
Posey.

 The film tells the story of the title character returning to Earth after a five-year absence.

 He finds that his love interest Lois Lane has moved on with her life, and that his archenemy Lex Luthor is plotting a scheme that will destroy Superman and the
world.
</item 3>

<item 4>
Lionel Luthor
Lionel Luthor is a fictional character portrayed by John Glover in the television series "Smallville".

 The character was initially a recurring guest in season one, and became a series regular in season two and continued until being written out of the show in
season seven.

 The character returned to the show in season ten again as a recurring guest role as a parallel universe (Earth-2) version of the character.

 In "Smallville", Lionel Luthor is the father of Lex Luthor (Michael Rosenbaum), and founder and CEO of LuthorCorp.

 Lex Luthor's father was first introduced in "Superman" comics by Jerry Siegel in 1961 and has since appeared in other Superman-related media under different
names.

 "Smallville" is the first appearance in which the character has been an intricate part of a Superman adaptation.

 Series developers Alfred Gough and Miles Millar created Lionel Luthor for "Smallville" to provide an antithesis to the parenting style of Jonathan Kent (John
Schneider) and Martha Kent (Annette O'Toole).
</item 4>

<item 5>
Harry Sweet
Harry Sweet (October 2, 1901 – June 18, 1933) was an American actor, director and screenwriter.

 He appeared in 57 films between 1919 and 1932.

 He also directed 54 films between 1920 and 1933, including one Harry Langdon short, two of the Tay Garnett- penned comedies Stan Laurel made for Joe Rock, and
fifteen of the earliest entries in the Edgar Kennedy "Average Man" series.
</item 5>

<item 6>
Lex Luthor: Man of Steel
Lex Luthor: Man of Steel (later collected as simply Luthor) is a five-issue monthly American comic book limited series written by Brian Azzarello and
illustrated by Lee Bermejo, which features Superman's nemesis Lex Luthor as the protagonist.
</item 6>

<item 7>
Superman: A Tale of Five Cities
Superman og Fredsbomben (translation: Superman and the Peace Bomb) or Superman: A Tale of Five Cities is a large prestige format 48-page graphic novel published
by Danish publisher Interpresse in 1990.

 In celebration of Superman's 50th anniversary in 1988, DC Comics incentived overseas publishers to produce an original story with the Man of Steel.

 Only the Interpresse editors from Denmark attended the call, comic book translator and writer Niels Søndergaard wrote the story featuring Clark Kent and Lois
Lane in a cold war adventure where Superman needs to devise a Lex Luthor plan to control all Nuclear Bombs in Europe.

 The storyline presents the Man of Steel visiting five capitals from northern Europe and Scandinavia: Amsterdam, Copenhagen, Oslo, Stockholm and Helsinki.

 The art and colors are provided by Teddy Kristiansen and the lettering by Rebecca Løwe.

 It was dedicated to danish editors Henning Kure and Ove Høyer, pioneers of the super-heroes comics in Denmark.
</item 7>

<item 8>
Alternative versions of Lex Luthor
The fictional character Lex Luthor has appeared in a number of media, always as the archenemy of Superman.

 Each version of the work typically establishes its own continuity, and sometimes introduces parallel universes, to the point where distinct differences in the
portrayal of the character can be identified.

 In addition, the DC Universe has been rewritten a number of times, establishing additional versions of the character.

 This article details and lists various versions of Lex Luthor depicted in works including DC Comics "Multiverse", "Elseworlds", television and film.
</item 8>

<item 9>
Lex Luthor (Smallville)
Lex Luthor is a fictional character from the television series "Smallville".

 He was a series regular from the pilot episode until the season seven finale, and has been played continuously by Michael Rosenbaum, with various actors
portraying the character as a child throughout the series.

 The character of Lex Luthor, first created for comic books by Jerry Siegel and Joe Shuster in 1940 as nemesis of Superman, was adapted to television in 2001 by
Alfred Gough and Miles Millar—this is only the third time the character has been adapted to a live action television series.

 The character has also appeared in various literature based on the "Smallville" television series, none of which directly continues from or into the television
episodes.
</item 9>

<item 10>
Gene Hackman
Eugene Allen "Gene" Hackman (born January 30, 1930) is a retired American actor and novelist.

 In a career spanning nearly five decades, Hackman was nominated for five Academy Awards, winning Best Actor in "The French Connection" and Best Supporting
Actor in "Unforgiven".

 He won three Golden Globes and two BAFTAs.
</item 10>
=== END ITEM id=7070eda2cf3e88a3 ===
