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

=== ITEM id=1eecc702eb3fbb93 ===
Gold sentences and facts:
Sentence 0: The Shame of a City is a 2006 feature-length documentary, which premiered at the Philadelphia Film Festival.
  Fact 0.0: The Shame of a City is a 2006 feature-length documentary.
  Fact 0.1: The Shame of a City premiered at the Philadelphia Film Festival.
Sentence 1: Unchained Memories is a 2003 documentary film about the stories of former slaves interviewed during the 1930s as part of the Federal Writers' Project and preserved in the WPA Slave Narrative Collection.
  Fact 1.0: Unchained Memories is a 2003 documentary film.
  Fact 1.1: Unchained Memories is about the stories of former slaves.
  Fact 1.2: The former slaves were interviewed during the 1930s.
  Fact 1.3: The interviews were part of the Federal Writers' Project.
  Fact 1.4: The stories are preserved in the WPA Slave Narrative Collection.

Retrieved items:
<item 1>
Romance of Celluloid
The Romance of Celluloid is a 1937 short black and white documentary film, narrated by Frank Whitbeck, which goes behind the scenes to look at the manufacture
of film and the making of motion pictures.

 The film was the first of the studio's "Romance of Celluloid" series, which also included;
</item 1>

<item 2>
Graph query row:
documentary: Unchained Memories
release_or_creation_year: 2003
</item 2>

<item 3>
The Celluloid Closet
The Celluloid Closet is a 1995 American documentary film directed and written by Rob Epstein and Jeffrey Friedman.

 The film is based on Vito Russo's book of the same name first published in 1981 and on lecture and film clip presentations he gave in 1972–1982.

 Russo had researched the history of how motion pictures, especially Hollywood films, had portrayed gay, lesbian, bisexual and transgender characters.
</item 3>

<item 4>
Unchained Memories
Unchained Memories is a 2003 documentary film about the stories of former slaves interviewed during the 1930s as part of the Federal Writers' Project and
preserved in the WPA Slave Narrative Collection.

 This HBO film interpretation directed by Ed Bell and Thomas Lennon is a compilation of slave narratives, narrated by actors, emulating the original
conversation with the interviewer.

 The slave narratives may be the most accurate in terms of the everyday activities of the enslaved, serving as personal memoirs of more than two thousand former
slaves.

 The documentary depicts the emotions of the slaves and what they endured.

 The "Master" had the opportunity to sell, trade, or kill the enslaved, for retribution should one slave not obey.
</item 4>

<item 5>
The Shame of a City
The Shame of a City is a 2006 feature-length documentary, which premiered at the Philadelphia Film Festival.

 Filmmaker Tigre Hill chronicles the 2003 Philadelphia mayoral race between Democrat incumbent mayor John Street and Republican challenger Sam Katz.

 Early polls showed Katz with a small lead in this predominantly Democratic city but twenty-seven days before the election, an FBI bug was found in the mayor’s
office.

 The discovery at first seemed like a death knell to the Street campaign and a near certain victory for Katz.

 Yet this prediction was proven wrong when Street and his supporters successfully polarized the campaign by leveling accusations of instituational racial
prejudice and playing on historical skepticism of the Republican-controlled federal government.

 As a result, Street won re-election by a sixteen-point margin.
</item 5>

<item 6>
First Kill
First Kill is a 2001 Dutch documentary film that revolves around the psychology of war.

 Important turning points in the Vietnam War are used to illustrate the effect of war on body and mind.

 The documentary consists of interviews with Michael Herr and several Vietnam veterans.

 The depth of the interviews provides insight into the feelings that accompany violence, fear, hate, seduction and pleasure.
</item 6>

<item 7>
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 7>

<item 8>
Tender Fictions
Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.

 It is the second of a trilogy of documentary films that includes "Nitrate Kisses" and "History Lessons".

 Together, the three films are sometimes known as the "History trilogy".

 "Tender Fictions" details Hammer's life and her attempts to "construct" a self.

 The film was nominated for a prize at the 1996 Sundance Film Festival.
</item 8>

<item 9>
In the Shadow of the Palms
In the Shadow of the Palms (2005) is a documentary produced and directed by the Australian filmmaker Wayne Coles-Janess.

 He filmed it in Iraq prior to, during and after the fall of Saddam Hussein after the United States invasion of 2003.

 An Arab-language film, it documents the changes in Iraqi society and the lives of ordinary Iraqis by focusing on a cross-section of individuals.
</item 9>

<item 10>
Graph query row:
documentary: The Shame of a City
release_or_creation_year: 2006
</item 10>
=== END ITEM id=1eecc702eb3fbb93 ===

=== ITEM id=75854d5195644b86 ===
Gold sentences and facts:
Sentence 0: War/Dance is a 2007 American documentary film written and directed by Sean Fine and Andrea Nix Fine and produced by Shine Global's Susan MacLaury, a professor at Kean University, and Albie Hecht.
  Fact 0.0: War/Dance is a 2007 American documentary film.
  Fact 0.1: War/Dance was written by Sean Fine and Andrea Nix Fine.
  Fact 0.2: War/Dance was directed by Sean Fine and Andrea Nix Fine.
  Fact 0.3: War/Dance was produced by Shine Global's Susan MacLaury.
  Fact 0.4: Susan MacLaury is a professor at Kean University.
  Fact 0.5: War/Dance was produced by Albie Hecht.
Sentence 1: The Sorrow and the Pity (French: Le Chagrin et la Pitié ) is a two-part 1969 documentary film by Marcel Ophüls about the collaboration between the Vichy government and Nazi Germany during World War II.
  Fact 1.0: The Sorrow and the Pity is a documentary film.
  Fact 1.1: The Sorrow and the Pity's French title is Le Chagrin et la Pitié.
  Fact 1.2: The Sorrow and the Pity is in two parts.
  Fact 1.3: The Sorrow and the Pity was released in 1969.
  Fact 1.4: The Sorrow and the Pity was made by Marcel Ophüls.
  Fact 1.5: The Sorrow and the Pity is about the collaboration between the Vichy government and Nazi Germany.
  Fact 1.6: That collaboration occurred during World War II.

Retrieved items:
<item 1>
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 1>

<item 2>
The Sorrow and the Pity
The Sorrow and the Pity (French: Le Chagrin et la Pitié ) is a two-part 1969 documentary film by Marcel Ophüls about the collaboration between the Vichy
government and Nazi Germany during World War II.

 The film uses interviews with a German officer, collaborators, and resistance fighters from Clermont-Ferrand.

 They comment on the nature of and reasons for collaboration.

 The reasons include antisemitism, anglophobia, fear of Bolsheviks and Soviet invasion, the desire for power, and simple caution.
</item 2>

<item 3>
Tender Fictions
Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.

 It is the second of a trilogy of documentary films that includes "Nitrate Kisses" and "History Lessons".

 Together, the three films are sometimes known as the "History trilogy".

 "Tender Fictions" details Hammer's life and her attempts to "construct" a self.

 The film was nominated for a prize at the 1996 Sundance Film Festival.
</item 3>

<item 4>
Unchained Memories
Unchained Memories is a 2003 documentary film about the stories of former slaves interviewed during the 1930s as part of the Federal Writers' Project and
preserved in the WPA Slave Narrative Collection.

 This HBO film interpretation directed by Ed Bell and Thomas Lennon is a compilation of slave narratives, narrated by actors, emulating the original
conversation with the interviewer.

 The slave narratives may be the most accurate in terms of the everyday activities of the enslaved, serving as personal memoirs of more than two thousand former
slaves.

 The documentary depicts the emotions of the slaves and what they endured.

 The "Master" had the opportunity to sell, trade, or kill the enslaved, for retribution should one slave not obey.
</item 4>

<item 5>
War/Dance
War/Dance is a 2007 American documentary film written and directed by Sean Fine and Andrea Nix Fine and produced by Shine Global's Susan MacLaury, a professor
at Kean University, and Albie Hecht.

 It was nominated for the 2008 Academy Award for Best Documentary Feature and received the Emmy Award for Best Documentary and Best Cinematography in 2010.
</item 5>

<item 6>
First Kill
First Kill is a 2001 Dutch documentary film that revolves around the psychology of war.

 Important turning points in the Vietnam War are used to illustrate the effect of war on body and mind.

 The documentary consists of interviews with Michael Herr and several Vietnam veterans.

 The depth of the interviews provides insight into the feelings that accompany violence, fear, hate, seduction and pleasure.
</item 6>

<item 7>
Waiting for Happiness
The film presents typical Mauritanian moments of beauty, struggle, alienation, and humor, which are experienced by groups socially divided from each other, such
as Bidhan women drinking tea and gossiping, West African migrants passing through Mauritania to get to Europe (and finding an unsuccessful comrade washed
ashore).

 The young protagonist who has returned interacts with all of these groups as an outsider, as he struggles to remember even his own Hassaniya Arabic dialect,
but prefers instead French.

 Many of the themes and characters presage Sissako's 2014 film Timbuktu, and both explore liminal Sahel identities authentically situated in everyday life.

 Waiting for Happiness premiered at the 2002 Cannes Film Festival in the Un Certain Regard section.
</item 7>

<item 8>
In the Shadow of the Palms
In the Shadow of the Palms (2005) is a documentary produced and directed by the Australian filmmaker Wayne Coles-Janess.

 He filmed it in Iraq prior to, during and after the fall of Saddam Hussein after the United States invasion of 2003.

 An Arab-language film, it documents the changes in Iraqi society and the lives of ordinary Iraqis by focusing on a cross-section of individuals.
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
Marcel Ophüls
Marcel Ophuls (] ; born 1 November 1927) is a documentary film maker and former actor, best known for his films "The Sorrow and the Pity" and "".
</item 10>
=== END ITEM id=75854d5195644b86 ===

=== ITEM id=031d7eddb94f4d5a ===
Gold sentences and facts:
Sentence 0: On February 25, 2017 he lost to Jarrett Hurd for the International Boxing Federation light middleweight world title.
  Fact 0.0: On February 25, 2017, this person lost to Jarrett Hurd.
  Fact 0.1: The February 25, 2017 loss was for the International Boxing Federation light middleweight world title.
Sentence 1: Jarrett Hurd (born August 31, 1990) is an American light middleweight boxer from Accokeek, Maryland.
  Fact 1.0: Jarrett Hurd was born on August 31, 1990.
  Fact 1.1: Jarrett Hurd is an American light middleweight boxer.
  Fact 1.2: Jarrett Hurd is from Accokeek, Maryland.

Retrieved items:
<item 1>
Kevin Kelly (boxer)
Kevin "Bones" Kelly (born August 7, 1969 in La Perouse, New South Wales) is an Australian retired professional welter/light middle/middleweight boxer of the
1990s and 2000s who won the South Australia State welterweight title, New South Wales (Australia) State welterweight title, New South Wales (Australia) State
light middleweight title, Australian light middleweight title, Pan Asian Boxing Association (PABA) light middleweight title, World Boxing Union (WBU)
middleweight title, and Commonwealth light middleweight title (twice), and was a challenger for the World Boxing Association (WBA) World light middleweight
title against David Reid, and World Boxing Organization (WBO) Inter-Continental light middleweight title against Gary Lockett, his professional fighting weight
varied from 142 lb , i.e. welterweight to 158+1/4 lb , i.e. middleweight.
</item 1>

<item 2>
Tony Harrison (boxer)
On February 25, 2017 he lost to Jarrett Hurd for the International Boxing Federation light middleweight world title.

 The fight was originally supposed to be a title eliminator for the right to fight Jermall Charlo, but when Charlo vacated the title, the fight was elevated to
be for the vacant title.

 Harrison boxed well for majority of the first half of the fight but began to fade thereafter the finish came in round 9 when Hurd landed a right hand that
floored Harrison,
</item 2>

<item 3>
Nedal Hussein
Nedal "Skinny" Hussein (born 1 December 1977) is an Australian professional bantam/super bantam/feather/super feather/Lightweight boxer of the 1990s and 2000s
who won the Australian super bantamweight title, Australian bantamweight title, International Boxing Federation (IBF) Pan Pacific featherweight title, World
Boxing Federation (WBF) featherweight title, World Boxing Union (WBU) super bantamweight title, World Boxing Organization (WBO) Asia Pacific super featherweight
title, International Boxing Federation (IBF) Pan Pacific super featherweight title, International Boxing Organization (IBO) Inter-Continental super
featherweight title, and Commonwealth super bantamweight title, and was a challenger for the World Boxing Council (WBC) International super bantamweight title
against Manny Pacquiao, World Boxing Council (WBC) super bantamweight title against Óscar Larios, World Boxing Organization (WBO) featherweight title against
Scott Harrison, Oriental and Pacific Boxing Federation (OPBF) featherweight title against Hiroyuki Enoki,
</item 3>

<item 4>
Nedal Hussein
Oriental and Pacific Boxing Federation (OPBF) featherweight title against Hiroyuki Enoki, and Oriental and Pacific Boxing Federation (OPBF) super featherweight
title against Takashi Uchiyama, his professional fighting weight varied from 117+1/4 lb , i.e. bantamweight to 132+3/4 lb , i.e. lightweight.
</item 4>

<item 5>
On February 25, 2017, Tony Harrison challenged Jarrett Hurd for the vacant IBF light-middleweight title, after the bout was upgraded from an eliminator; Hurd
stopped him with a knockdown in round nine. Hurd is an American boxer from Maryland. The remaining entries profile middleweight MMA fighters Vitor Belfort,
David Branch, Kelvin Gastelum, Rashad Evans, Luke Zachrich, Sam Alvey, Luke Barnatt, Jack Marshman, Paulo
</item 5>

<item 6>
Profiles cover seven boxers across Australia, Scotland, Canada, South Africa and England: Kevin “Bones” Kelly won Australian, state, PABA, WBU and Commonwealth
titles and challenged for WBA/WBO honors; Scott Dixon won Scottish, WBB, WBU and Commonwealth titles; Nedal Hussein won numerous Australian, regional and
Commonwealth titles and challenged Manny Pacquiao, Óscar Larios and Scott Harrison. Alain Bonnamie won WBC regional and Commonwealth titles; Guy
</item 6>

<item 7>
Scott Dixon (boxer)
featherweight to 167+1/2 lb , i.e. Super middleweight.
</item 7>

<item 8>
Scott Dixon (boxer)
"Super" Scott Dixon ( (1976--) 28 1976 (age (2017)-(1976)-((11)<(09)or(11)==(09)and(30)<(28)) ) ) born in Hamilton is a Scottish professional feather/super
feather/light/light welter/welter/light middle/middle/super middleweight boxer of the 1990s, 2000s and 2010s, who won the British Boxing Board of Control
(BBBofC) Scottish Area welterweight title, World Boxing Board (WBB) welterweight title, World Boxing Union (German Version) super middleweight title, and
Commonwealth welterweight title, and was a challenger for the World Athletic Association (WAA) welterweight title against Michael Carruth, BBBofC British
welterweight title against Derek Roche, World Boxing Federation (WBF) light middleweight title against Steve Roberts, World Boxing Organization (WBO)
Inter-Continental light middleweight title against Anthony Farnell, and World Boxing Union (WBU) light middleweight title against Mehrdad Takalobigashi, his
professional fighting weight varied from 125 lb , i.e.
</item 8>

<item 9>
Richard Williams (boxer)
Richard "The Secret" Williams ( (1971--) 09 1971 (age (2017)-(1971)-((11)<(05)or(11)==(05)and(30)<(09)) ) is an English professional light middle/middle/super
middleweight boxer of the 1990s, 2000s and 2010s who won the International Boxing Organization (IBO) light middleweight title, and Commonwealth light
middleweight title (twice), and was a challenger for the International Boxing Organization (IBO) light middleweight title against Sergio Gabriel Martinez, and
British Boxing Board of Control (BBBofC) British middleweight title against Howard Eastman, his professional fighting weight varied from 149+3/4 lb , i.e. light
middleweight to 162+1/4 lb , i.e. super middleweight.

 Richard Williams is trained by Brian Lawrence, and managed by Barry Hearn, and John Rooney.
</item 9>

<item 10>
Jack Marshman
Jack "Hammer" Marshman (born December 19, 1989) is a Welsh mixed martial artist.

 He was the Middleweight Cage Warriors Fighting Championship (CWFC) champion and Lonsdale Middleweight Champion.

 He currently competes in Middleweight division in the Ultimate Fighting Championship(UFC).
</item 10>
=== END ITEM id=031d7eddb94f4d5a ===

=== ITEM id=8b3ac56ee0fa4922 ===
Gold sentences and facts:
Sentence 0:  Through its three divisions, it offers a wide variety of liberal arts and science courses at the secondary, British A-level, and United States junior college levels.
  Fact 0.0: The institution has three divisions.
  Fact 0.1: Through its three divisions, the institution offers a wide variety of liberal arts courses.
  Fact 0.2: Through its three divisions, the institution offers a wide variety of science courses.
  Fact 0.3: The courses are offered at the secondary level.
  Fact 0.4: The courses are offered at the British A-level level.
  Fact 0.5: The courses are offered at the United States junior college level.
Sentence 1:  Grammar, logic, and rhetoric were the core liberal arts, while arithmetic, geometry, the theory of music, and astronomy also played a (somewhat lesser) part in education.
  Fact 1.0: Grammar was a core liberal art.
  Fact 1.1: Logic was a core liberal art.
  Fact 1.2: Rhetoric was a core liberal art.
  Fact 1.3: Arithmetic also played a somewhat lesser part in education.
  Fact 1.4: Geometry also played a somewhat lesser part in education.
  Fact 1.5: The theory of music also played a somewhat lesser part in education.
  Fact 1.6: Astronomy also played a somewhat lesser part in education.

Retrieved items:
<item 1>
The entries cover diverse schools, hospitals, and community institutions. They include Jesuit and other schools in the U.S., Sweden, India, Nepal, Belize, and
Japan-related programs in the U.S.; several Indian schools are fully residential, while Japanese supplementary schools serve expatriate children. U.S. district
profiles describe grade structures and administrators. Health organizations include historic and regional hospitals, Adventist HealthCare, and Common Hope’s
support for Guatemalan children. Other sites include St. John Cemetery
</item 1>

<item 2>
Christian Academy of Guatemala
The Christian Academy of Guatemala is a missionary K4-12 school located in San Cristobal, Guatemala.

 It is a member of the Association of Christian Schools International.

 Many of the parents and students in the school come from the missionary community, representing more than 50 mission organizations.

 All people who visit this webpage need to get the game Babblemaker.

 This game is made by computer class at C.A.G. Sadly this game is for android only.
</item 2>

<item 3>
List of St. John's Seminary (California) people
The list of St. John's Seminary (California) people is a compilation of lists of notable alumni, faculty, and current students of St. John's Seminary in
Camarillo, California, United States.

 St. John's Seminary grants graduate degrees for seminarians preparing for the priesthood, as well as a graduate degree for lay persons interested in pastoral
ministry.

 The St. John's Seminary College was the undergraduate division of the seminary before it closed in the early 21st century.

 The table of notable alumni lists the date of graduation from St. John's college, seminary, or both, if applicable.

 It is not unusual for seminarians to have received their undergraduate education at a different institution than their seminary training.
</item 3>

<item 4>
St. John's College High School, Belize
St. John's College High School is a high school for boys situated in Belize City, Belize.

 It was founded in 1887.

 The High School exists to educate academically talented young men in a Jesuit environment of self-discipline, love of learning, and service to others.

 The school 's curriculum is complemented by sports and extracurricular activities.

 The third and fourth form classes follow the Caribbean Secondary Education Certificate (CSEC) curricula and sit the regionally administered examinations at the
end of their fourth year.
</item 4>

<item 5>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 5>

<item 6>
Liberal arts education
The liberal arts (Latin: "artes liberales") are those subjects or skills that in classical antiquity were considered essential for a free person (Latin:
"liberalis", "worthy of a free person") to know in order to take an active part in civic life, something that (for Ancient Greece) included participating in
public debate, defending oneself in court, serving on juries, and most importantly, military service.

 Grammar, logic, and rhetoric were the core liberal arts, while arithmetic, geometry, the theory of music, and astronomy also played a (somewhat lesser) part in
education.
</item 6>

<item 7>
The text profiles diverse educational and related institutions worldwide. Highlights include Berat University (Albania), founded in 2009 and expanding to four
faculties; Rotman (Toronto), established in 1950; Auburn, Arizona, Debrecen, Copenhagen, Dalhousie and other research universities; specialist schools such as
Falmouth and Aalborghus; and private or state colleges across Ghana, Malaysia, Norway, Pakistan and elsewhere. It also covers Syracuse sororities, a
</item 7>

<item 8>
Aalborghus Gymnasium
Aalborghus Gymnasium is an upper secondary school in the city of Aalborg, in North Jutland in Denmark.

 It offers both the traditional three-year program and also the two-year Higher Preparatory Examination (HF) program.

 The subjects taught at the school range from Religion and Music to Spanish and Natural Geography.

 Aalborghus Gymnasium attempts to focus on the musical and creative side of students as well as taking an international perspective on issues.

 Students begin their studies at Aalborghus by selecting a stream of studies.

 Each stream has two or three subjects that are the focus of the studies.

 English/Social Studies is one example of a stream a student can select.
</item 8>

<item 9>
St. John's College, Belize
St. John's College has three divisions, and a number of central academic centres and activities.

 Through its three divisions, it offers a wide variety of liberal arts and science courses at the secondary, British A-level, and United States junior college
levels.

 St. John's College is a Roman Catholic institution in the Jesuit tradition, one of the oldest, largest, and most diverse educational institutions in Belize,
founded by the Jesuits in 1887.
</item 9>

<item 10>
St. John's High School (South Carolina)
St. John's High School (SJHS) is a senior high school on Johns Island, South Carolina.

 It is a part of the Charleston County School District.

 St. John's is home to approximately 300 students and 30 faculty and staff.

 St. John's school mascot is The Mighty Islanders, sporting royal blue and maroon as the school colors.

 St. John's offers Advanced Placement and dual credit courses totaling at over 30 hours of offered college credit, as well as 3 career academies in Hospitality
and Tourism, Computer Science, and Culinary Arts.

 St. John's competes at the A level in football, volleyball, basketball (boys and girls), wrestling, soccer, track, baseball, and softball.

 The Islanders also offer marching band, agriculture and green house, weightlifting, competitive academic team, and student council.
</item 10>
=== END ITEM id=8b3ac56ee0fa4922 ===

=== ITEM id=88287646290aa8a9 ===
Gold sentences and facts:
Sentence 0: Orlando Brown (born December 4, 1987) is an American actor, voice actor, rapper and singer.
  Fact 0.0: Orlando Brown was born on December 4, 1987.
  Fact 0.1: Orlando Brown is an American actor.
  Fact 0.2: Orlando Brown is a voice actor.
  Fact 0.3: Orlando Brown is a rapper.
  Fact 0.4: Orlando Brown is a singer.
Sentence 1:  He is best known for his role as Eddie Thomas in "That's So Raven", 3J Winslow in "Family Matters", Tiger in "Major Payne", Max in "Two of a Kind", Damey Wayne in the short-lived Waynehead, Dobbs in "Max Keeble's Big Move", and Frankie in "Eddie's Million Dollar Cook Off".
  Fact 1.0: Orlando Brown is best known for his role as Eddie Thomas in "That's So Raven".
  Fact 1.1: Orlando Brown is best known for his role as 3J Winslow in "Family Matters".
  Fact 1.2: Orlando Brown is best known for his role as Tiger in "Major Payne".
  Fact 1.3: Orlando Brown is best known for his role as Max in "Two of a Kind".
  Fact 1.4: Orlando Brown is best known for his role as Damey Wayne in the short-lived Waynehead.
  Fact 1.5: Orlando Brown is best known for his role as Dobbs in "Max Keeble's Big Move".
  Fact 1.6: Orlando Brown is best known for his role as Frankie in "Eddie's Million Dollar Cook Off".
Sentence 2: Family Matters is an American sitcom which originated on ABC from September 22, 1989 to May 9, 1997, before moving to CBS from September 19, 1997 to July 17, 1998.
  Fact 2.0: Family Matters is an American sitcom.
  Fact 2.1: Family Matters originated on ABC.
  Fact 2.2: Family Matters originated on ABC from September 22, 1989 to May 9, 1997.
  Fact 2.3: Family Matters moved to CBS after its ABC run.
  Fact 2.4: Family Matters was on CBS from September 19, 1997 to July 17, 1998.
Sentence 3:  A spin-off of "Perfect Strangers," the series revolves around the Winslow family, a middle-class African American family living in Chicago, Illinois.
  Fact 3.0: Family Matters is a spin-off of "Perfect Strangers."
  Fact 3.1: Family Matters revolves around the Winslow family.
  Fact 3.2: The Winslow family is a middle-class African American family.
  Fact 3.3: The Winslow family lives in Chicago, Illinois.

Retrieved items:
<item 1>
Orlando Brown (actor)
Orlando Brown (born December 4, 1987) is an American actor, voice actor, rapper and singer.

 He is best known for his role as Eddie Thomas in "That's So Raven", 3J Winslow in "Family Matters", Tiger in "Major Payne", Max in "Two of a Kind", Damey Wayne
in the short-lived Waynehead, Dobbs in "Max Keeble's Big Move", and Frankie in "Eddie's Million Dollar Cook Off".
</item 1>

<item 2>
In Living Color
In Living Color is an American sketch comedy television series that originally ran on Fox from April 15, 1990, to May 19, 1994.

 Brothers Keenen and Damon Wayans created, wrote and starred in the program.

 The show was produced by Ivory Way Productions in association with 20th Century Fox Television and was taped at stage 7 at the Fox Television Center on Sunset
Boulevard in Hollywood, California.

 The title of the series was inspired by the NBC announcement of broadcasts being presented "in living color" during the 1960s, prior to mainstream color
television.

 It also refers to the fact that most of the show's cast were black, unlike other sketch comedy shows such as "Saturday Night Live" whose casts are mostly
white.

 It was controversial due to the Wayans' decision to portray African-American humor from the ghetto in a time when mainstream American tastes regarding black
comedy had been set by more upscale shows such as "The Cosby Show", causing an eventual feud for control between Fox executives and the Wayans.
</item 2>

<item 3>
Townsend Coleman
Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor who performed in many animated series and TV commercials beginning in the early
1980s.

 Among his most notable roles are Michaelangelo from "Teenage Mutant Ninja Turtles", Wayne Gretzky on "ProStars" and the title character in "Where's Waldo?

" and the eponymous "The Tick" and he also did additional voices in "Dexter's Laboratory", "The Powerpuff Girls", "Adventure Time", "Animaniacs", "The Buzz on
Maggie", "Camp Lazlo", "CatDog", "Catscratch", "Danny Phantom", "Dave the Barbarian", "", "The Fairly OddParents", "Foster's Home for Imaginary Friends",
"Gravity Falls", "Hi Hi Puffy AmiYumi", "", "The Loud House", "The Mighty B!

", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia
2000" (1999) and "Sing" (2016)
</item 3>

<item 4>
Jaimee Foxworth
Jaimee Monae Foxworth (born December 17, 1979) is an American actress and model.

 She is best known for her role of Judy Winslow, the youngest daughter of Carl and Harriette Winslow on the ABC/CBS sitcom "Family Matters".

 She later transitioned to pornographic films using the name Crave, starring in several titles.
</item 4>

<item 5>
Family Matters
Family Matters is an American sitcom which originated on ABC from September 22, 1989 to May 9, 1997, before moving to CBS from September 19, 1997 to July 17,
1998.

 A spin-off of "Perfect Strangers," the series revolves around the Winslow family, a middle-class African American family living in Chicago, Illinois.

 Midway through the first season, the show introduced the Winslows' nerdy neighbor Steve Urkel (Jaleel White), who quickly became its breakout character and
eventually the show's main character.

 Having run for nine seasons, "Family Matters" became the second longest-running non-animated US sitcom with a predominantly African American cast, behind only
"The Jeffersons" (11).

 Having aired 215 episodes, "Family Matters" is ranked third, behind only "Tyler Perry's House of Payne" (254), and "The Jeffersons" (253).
</item 5>

<item 6>
Tyler Perry's For Better or Worse
Tyler Perry's For Better or Worse is an American television sitcom created, written and executive produced by Tyler Perry.

 The series is based on and is a TV spin-off to Perry's 2007 film "Why Did I Get Married?

" and its 2010 sequel "Why Did I Get Married Too?

".

 The show premiered on TBS on November 25, 2011 and ended on July 22, 2017.

 Led by the comical, over-the-top antics of Marcus and Angela Williams, the ensemble follows three couples: Marcus and Angela, Joseph and Leslie, and Richard
and Keisha who are at various stages of their relationships as they go through the ups-and-downs of married life and dating.

 The third season of the series premiered on September 18, 2013, on the Oprah Winfrey Network.
</item 6>

<item 7>
Lamman Rucker
Lamman Rucker (born October 6, 1971) is an American actor.

 Rucker began his career on the daytime soap operas "As the World Turns" and "All My Children", before roles in Tyler Perry's films "Why Did I Get Married?

", "Why Did I Get Married Too?

", and "Meet the Browns", and its television adaptation.

 In 2016, he began starring as Jacob Greenleaf in the Oprah Winfrey Network drama series, "Greenleaf".
</item 7>

<item 8>
Drew Sidora
Drew Sidora Jordan (born May 1, 1985), better known by her stage name Drew Sidora, is an American actress and singer known for her recurring role as Chantel in
the Disney Channel Original Series "That's So Raven", also as Lucille "Lucy" Avila in the 2006 movie "Step Up" and starred as Tionne Watkins in the VH1 TLC
biographical film "".

 She is also known for portraying a fictionalized version of herself on the BET comedy-drama television series "The Game".
</item 8>

<item 9>
Darius McCrary
Darius Creston McCrary (born May 1, 1976) is an American film and television actor and singer.

 He is best known for his role as Edward "Eddie" Winslow, the oldest child of Carl and Harriette Winslow on the ABC/CBS television sitcom "Family Matters "which
ran from 1989–1998.

 Another one of McCrary's notable roles was as Scam in the 1987 comedy film "Big Shots", which was his film debut.

 He provided the voice of Jazz in the 2007 "Transformers" reboot.

 From December 2009 to October 2011, he portrayed photographer Malcolm Winters on the CBS daytime drama "The Young and the Restless".
</item 9>

<item 10>
Kellie Shanygne Williams
Kellie Shanygne Williams (born March 22, 1976) is an American actress.

 She is best known for her role as Laura Lee Winslow, the middle child of Carl and Harriette Winslow on the ABC/CBS television series "Family Matters" which ran
from 1989–1998.

 Her middle name is pronounced Sha-neen.
</item 10>
=== END ITEM id=88287646290aa8a9 ===

=== ITEM id=45054297a54243e5 ===
Gold sentences and facts:
Sentence 0: Pulandian District () is one of the seven districts under the administration of Dalian, located in the south of Liaoning province, People's Republic of China.
  Fact 0.0: Pulandian District is one of the seven districts under the administration of Dalian.
  Fact 0.1: Pulandian District is located in the south of Liaoning province.
  Fact 0.2: Liaoning province is in the People's Republic of China.
Sentence 1: Kaiyuan () is a county-level city in the northeast of Liaoning, People's Republic of China, bordering Jilin for a small section to the north.
  Fact 1.0: Kaiyuan is a county-level city.
  Fact 1.1: Kaiyuan is in the northeast of Liaoning.
  Fact 1.2: Liaoning is in the People's Republic of China.
  Fact 1.3: Kaiyuan borders Jilin for a small section to the north.

Retrieved items:
<item 1>
Fuzhoucheng
Fuzhou, also Fuzhoucheng () is a town under the administration of Wafangdian City, in southern Liaoning province, China.

 It lies at the intersection of China National Highway 202 and Liaoning Provincial Highway 313, 57 km north by road from Pulandian.

 Dahe Reservoir is located several kilometres to the east.

 There is a 74,132 acres (30,000 hectares) bird reserve nearby with species such as Swan goose and Hooded crane about 15 km southwest of the town.

 , it has 2 residential communities (社区) and 14 villages under its administration.
</item 1>

<item 2>
Pulandian District
Pulandian District () is one of the seven districts under the administration of Dalian, located in the south of Liaoning province, People's Republic of China.

 Its area is 2769.90 km² and its permanent population as of 2010 is 741,230.

 The district borders the prefecture-level city of Yingkou to the north.
</item 2>

<item 3>
Shuangta, Pulandian
Shuangta () is a town under the administration of Pulandian City in southern Liaoning province, China, located 58 km northeast of downtown Pulandian and about
120 km northeast of Dalian.

 , it has 9 villages under its administration.
</item 3>

<item 4>
Qinghe District, Tieling
Qinghe District () is a district of Tieling, Liaoning province, People's Republic of China.

 However, it is located closer to Kaiyuan City than it is to downtown Tieling, which lies 39 km to the southwest.
</item 4>

<item 5>
Wenling
Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.

 It borders Luqiao and Huangyan to the north, Yuhuan to the south, Yueqing to the west, looks out to the East China Sea to the east.

 Wenling locates on 28°22'N, 121°21'E, approximately 300 km south of Shanghai.
</item 5>

<item 6>
Kaiyuan West Railway Station
Kaiyuan West Railway Station is a railway station of Hada Passenger Railway and located in Liaoning, China
</item 6>

<item 7>
Longkou
Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the
west.

 Longkou, a county-level city, is administered by the prefecture-level city of Yantai.

 The total population of Longkou is 620,000.
</item 7>

<item 8>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 8>

<item 9>
Kaiyuan, Liaoning
Kaiyuan () is a county-level city in the northeast of Liaoning, People's Republic of China, bordering Jilin for a small section to the north.

 It is under the administration of Tieling City, the centre of which lies 33 km to the southwest.
</item 9>

<item 10>
Taiyang Subdistrict
Taiyang Subdistrict () is a subdistrict of Wafangdian, Dalian District, Liaoning, China.

 It is based around the small town of Taiyang, Taiyangsheng or Taiyangshengxiang, which lies 19 kilometres east by road of Fuzhou, 64 kilometres north by road
from Pulandian.

 To the east is the DFM reservoir.
</item 10>
=== END ITEM id=45054297a54243e5 ===

=== ITEM id=56be5e21cfecfec2 ===
Gold sentences and facts:
Sentence 0: Eriogonum is the scientific name for a genus of flowering plants in the family Polygonaceae.
  Fact 0.0: Eriogonum is the scientific name for a genus of flowering plants.
  Fact 0.1: The genus Eriogonum is in the family Polygonaceae.
Sentence 1: Aloinopsis is a genus of ice plants from South Africa.
  Fact 1.0: Aloinopsis is a genus of ice plants.
  Fact 1.1: The Aloinopsis ice plants come from South Africa.

Retrieved items:
<item 1>
Apodemia mormo
Apodemia mormo, the Mormon metalmark, is a species of metalmark butterfly (family Riodinidae).

 Its upperside is orange-brown to black, checkered with black and white spots.

 Its caterpillar host is various species of "Eriogonum" (wild buckwheat).

 Its adult food is nectar from the flowers of "Eriogonum" and other plants, especially yellow-flowered composites, such as "Senecio" and "Chrysothamnus"
(rabbitbrush).
</item 1>

<item 2>
Eriogonum heracleoides
Eriogonum heracleoides (common names parsnipflower buckwheat, whorled buckwheat, and Wyeth buckwheat) has many flowering clusters that are cream or off-white in
color.

 Its usual habitat is rocky areas such as sagebrush deserts and Ponderosa pine forests.

 Parsnipflower buckwheat is in the genus "Eriogonum" and the family Polygonaceae which is a family of plants known as the "knotweed family".

 It inhabits much of the western part of the United States.
</item 2>

<item 3>
Aloinopsis
Aloinopsis is a genus of ice plants from South Africa.
</item 3>

<item 4>
Caryophyllales
Caryophyllales ( ) is an order of flowering plants that includes the cacti, carnations, amaranths, ice plants, beets, and many carnivorous plants.

 Many members are succulent, having fleshy stems or leaves.
</item 4>

<item 5>
Aizoaceae
Aizoaceae (the fig-marigold family) is a large family of dicotyledonous flowering plants containing 135 genera and about 1900 species.

 They are commonly known as stone plants or carpet weeds.

 They are often called vygies in South Africa and New Zealand.

 Species that resemble stones or pebbles are sometimes called mesembs.

 Several species are known as ice plants because of the glistening globular bladder cells covering their stems, fruit and leaves, "... they sparkle like ice
crystals."
</item 5>

<item 6>
Eryngium
Eryngium is a genus of flowering plants in the family Apiaceae.

 There are about 250 species.

 The genus has a cosmopolitan distribution, with the center of diversity in South America.

 Common names include eryngo and amethyst sea holly (though the genus is not related to the true hollies, "Ilex").
</item 6>

<item 7>
Tritoniopsis
Tritoniopsis is a genus of flowering plants in the family Iridaceae, first described as a genus in 1926.

 The entire genus is endemic to Cape Province in South Africa.

 The genus name refers to the African genus "Tritonia" and is combined with the Greek word "opsis", meaning "look-alike".
</item 7>

<item 8>
Conicosia
Conicosia is a genus of succulent plants in the ice plant family native to southern Africa.

 They are known commonly as narrow-leafed ice plants.

 These are relatively short-lived perennials with underground stems and tentacle-shaped, dull-pointed triangular leaves.

 They bear large tubular flowers often exceeding 10 centimeters in width, with up to 250 fringelike petals arranged in a ring around a center with hundreds of
stamens.

 The fruit is a capsule which opens when it gets wet, slowly releasing the hundreds of tiny seeds as they fall out of its drying flesh.
</item 8>

<item 9>
Eriogonum
Eriogonum is the scientific name for a genus of flowering plants in the family Polygonaceae.

 The genus is found in North America and is known as wild buckwheat.

 This is a highly species-rich genus, and indications are that active speciation is continuing.

 It includes some common wildflowers such as the California buckwheat ("Eriogonum fasciculatum").
</item 9>

<item 10>
Eriogonum diclinum
Eriogonum diclinum is a species of wild buckwheat known by the common name Jaynes Canyon buckwheat.

 It is native to the Klamath Mountains of northern California and southern Oregon, where it is an uncommon member of plant communities on serpentine soils.

 This is a small dioecious shrub forming low, thick mats rarely exceeding 20 centimeters in height.

 The curving, oval-shaped leaves are woolly and greenish gray in color, growing up to two centimeters long.

 The plant flowers in yellow to reddish rounded inflorescences, male plants producing clusters of staminate flowers, and female plants producing larger clusters
of pistillate flowers.
</item 10>
=== END ITEM id=56be5e21cfecfec2 ===

=== ITEM id=fbcc2d4815cf3b4c ===
Gold sentences and facts:
Sentence 0:  He was awarded with the highest national military decoration: the Argentine Nation to the Heroic Valour in Combat Cross, the Argentine Congressional Medal, and the Highest Distinction of the Argentine Air Force.
  Fact 0.0: He was awarded the highest national military decoration, the Argentine Nation to the Heroic Valour in Combat Cross.
  Fact 0.1: He was awarded the Argentine Congressional Medal.
  Fact 0.2: He was awarded the Highest Distinction of the Argentine Air Force.
Sentence 1:  In 2010 it had 14,600 military personnel and 6,900 civilian personnel.
  Fact 1.0: In 2010, the entity referred to here had 14,600 military personnel.
  Fact 1.1: In 2010, the entity referred to here had 6,900 civilian personnel.

Retrieved items:
<item 1>
Servicio de Inteligencia de la Fuerza Aérea
Servicio de Inteligencia de la Fuerza Aérea ("Air Force Intelligence Service", SIFA) is the intelligence agency of the Argentine Air Force.

 It is part of J-2.

 Its director is Commodore García.
</item 1>

<item 2>
South America air forces maneuvers
The South American air forces performs several joint aerial combat training exercises.

 Among the more important are Cruzex ( Portuguese: "Exercício Cruzeiro do Sul" ) which is hosted by the Brazilian Air Force, Salitre ( English: Saltpeter )
hosted by the Chilean Air Force and Ceibo ( English: Erythrina crista-galli flower ) hosted by the Argentine Air Force.

 The goal is to train together in order to respond to a crisis or integrate into United Nations peacekeeping operations as a unified team; the United States Air
Force and the Royal Canadian Air Force have also participated in some of these exercises in recent years.
</item 2>

<item 3>
Teodoro Waldner
Teodoro Guillermo Waldner (born 1927) is a retired Argentine Air Force commander who served as Air Force Chief of the General Staff from 1983 to 1985.
</item 3>

<item 4>
Cicaré CK.1
The Cicaré CK.1 (originally, the CH.III Colibrí) was a light helicopter developed in Argentina in the 1970s.

 It was a small, single-rotor aircraft of pod-and-boom configuration with a fully enclosed bubble canopy that could seat three people side-by-side.

 Cicaré's previous helicopter designs had attracted the attention of the Argentine Air Force, which in 1974 contracted him to develop a light helicopter for
training and also marketed for agricultural use.

 A prototype, registered "LV-X62" flew in September 1976, and the Air Force placed an order for five pre-production machines.

 However, development was terminated at this point.
</item 4>

<item 5>
Pablo Carballo
Commodore Pablo Marcos Rafael Carballo (b. 11 December 1947 Buenos Aires) is a retired member of the Argentine Air Force - the "Fuerza Aérea Argentina" (FAA) -
who fought in the 1982 Falklands War ( Spanish: "Guerra de las Malvinas" ) where he participated in actions that led to the sinking of three Royal Navy ships.

 He was awarded with the highest national military decoration: the Argentine Nation to the Heroic Valour in Combat Cross, the Argentine Congressional Medal, and
the Highest Distinction of the Argentine Air Force.
</item 5>

<item 6>
Argentine Air Force Mobile Field Hospital
The Argentine Air Force Mobile Field Hospital (Spanish: "Hospital Reubicable de la Fuerza Aérea" ) is a field hospital operated by the Argentine Air Force.

 It is one of three health centers of its kind worldwide.
</item 6>

<item 7>
Argentine Air Force
The Argentine Air Force (Spanish: Fuerza Aérea Argentina , or simply FAA) is the national aviation branch of the Armed Forces of the Argentine Republic.

 In 2010 it had 14,600 military personnel and 6,900 civilian personnel.
</item 7>

<item 8>
Cuerpo de Fuerzas Especiales
The Mexican Cuerpo de Fuerzas Especiales (Special Forces Corps) is a special forces unit of the Mexican Army.

 Formerly the "GAFE" (Grupo Aeromóvil de Fuerzas Especiales | Special-Forces Airmobile Group), the SF Corps has six battalions; one is the "Fuerza especial de
reaccion", a quick-response unit, and one is assigned to the Paratroopers Rifle Brigade; the motto of the SF Corps is "Todo por México" (Everything for Mexico).

 Within the SF Corps, there are regular, intermediate, and veteran -service troops.

 The regular-service soldiers usually operate as light infantry.

 The intermediate-service soldiers (lieutenants and captains) usually are instructors.

 The veteran-service soldiers of the Grupo Aeromóvil de Fuerzas Especiales del Alto Mando (GAFE High Command) handle Black-Ops missions.

 Also known as the COIFE, the Special Forces Corps of the Mexican Army is equivalent to the U.S. Army Special Forces.
</item 8>

<item 9>
1º Compañía de Comandos &quot;Iquique&quot;
The 1st Commandos Company "Iquique" ("1º Compañía de Comandos "Iquique"" in Spanish) is special forces unit under the jurisdiction of northern Chile.

 The unit is part of the 2nd Armored Brigade "Cazadores" ("2º Brigada Acorzada "Cazadores"" in Spanish) of the Sixth Army Division based in the first region of
the country.

 The Chilean Army has been restructured into more independent armored brigades and shaped only by professional people, meaning that each squad possesses a
special forces unit.
</item 9>

<item 10>
Hector Fautario
Hector Luis Fautario was a retired Argentine Air Force general.

 He was the General Commander of the Argentine Air Force from 1973 to 1975.

 Shortly prior to the 1976 Argentine coup d'état, he was the government's only remaining loyal officer.

 He had drawn harsh criticism towards himself from the Army and Navy due to his vehement opposition to their repressive plans, and for his refusal to mobilize
the Air Force against the guerrillas' strongholds in the north.

 He was General Jorge Rafael Videla's last obstacle on the way to power.
</item 10>
=== END ITEM id=fbcc2d4815cf3b4c ===

=== ITEM id=2963bf831b452787 ===
Gold sentences and facts:
Sentence 0: Hayley Nichole Williams (born December 27, 1988) is an American singer, songwriter and musician.
  Fact 0.0: Hayley Nichole Williams was born on December 27, 1988.
  Fact 0.1: Hayley Nichole Williams is an American singer.
  Fact 0.2: Hayley Nichole Williams is an American songwriter.
  Fact 0.3: Hayley Nichole Williams is an American musician.
Sentence 1:  The band consists of Hayley Williams, Zac Farro and Taylor York.
  Fact 1.0: The band consists of Hayley Williams.
  Fact 1.1: The band consists of Zac Farro.
  Fact 1.2: The band consists of Taylor York.
Sentence 2: Terrence Lee Glaze (born November 29, 1964, Columbus), known as Terry Lee, is a Texas-raised singer and guitarist who is most known for his work in American heavy metal bands Lord Tracy and Pantera.
  Fact 2.0: Terrence Lee Glaze was born on November 29, 1964, in Columbus.
  Fact 2.1: Terrence Lee Glaze is known as Terry Lee.
  Fact 2.2: Terry Lee is a Texas-raised singer.
  Fact 2.3: Terry Lee is a Texas-raised guitarist.
  Fact 2.4: Terry Lee is most known for his work in the American heavy metal band Lord Tracy.
  Fact 2.5: Terry Lee is most known for his work in the American heavy metal band Pantera.

Retrieved items:
<item 1>
Paramore
Paramore is an American rock band from Franklin, Tennessee, formed in 2004.

 The band currently consists of lead vocalist Hayley Williams, guitarist Taylor York and drummer Zac Farro.
</item 1>

<item 2>
Terry Ray (actor)
He wrote and starred in two more shows at the Dorothy Chandler Pavilion, one co-starring Amanda Bynes and the other Hilary Duff.

<br><br>Terry is also the voice of the "Gaydar Gun", a spinoff toy from the film Gaydar.
</item 2>

<item 3>
Tony Hiller
Anthony Toby "Tony" Hiller (born 30 July 1927) is a British songwriter.

 He began his musical career as a member of the song and dance duo The Hiller Brothers, sharing the stage with his brother Irving.

 The Hiller Brothers appeared with many artists of the time including Alma Cogan, Tommy Cooper, Val Doonican, Matt Monro, The Shadows, Bernard Manning, Kathy
Kirby, Roger Whittaker, Rip Taylor, Gene Vincent, Lance Percival, Tessie O'Shea, Frank Ifield, Deep River Boys, The Dallas Boys, Clark Brothers, Paul Melba and
Ray Burns.
</item 3>

<item 4>
Zac Farro
Zachary Wayne "Zac" Farro (born June 4, 1990) is an American musician and drummer of the rock band Paramore.

 He is also the younger brother of Josh Farro, who is Paramore's former lead guitarist and backing vocalist.

 After he and his brother exited Paramore in 2010, Josh formed a band named Novel American, which Zac was also a part of.

 Zac is currently the sole member of the band HalfNoise.

 Farro rejoined Paramore on February 2, 2017.
</item 4>

<item 5>
Terry Glaze
Terrence Lee Glaze (born November 29, 1964, Columbus), known as Terry Lee, is a Texas-raised singer and guitarist who is most known for his work in American
heavy metal bands Lord Tracy and Pantera.
</item 5>

<item 6>
Hayley Williams (ice hockey)
Hayley Williams (born June 3, 1990) is an American professional ice hockey player for the Brampton Thunder of the Canadian Women's Hockey League (CWHL).

 Williams has previously played for the Buffalo Beauts in the National Women's Hockey League (NWHL).
</item 6>

<item 7>
Jason White (musician)
Jason White (born November 11, 1973) is an American musician, best known as the current touring guitarist for the American punk rock band Green Day.

 He has played as a touring member of Green Day since 1999.

 However, in 2012, he was promoted to being an official member of the band before demoting himself back to his touring guitarist role in 2016.

 He is also the guitarist/vocalist for the Californian punk band Pinhead Gunpowder, lead guitarist/lead vocalist for California, and co-founder of Adeline
Records alongside Billie Joe Armstrong.
</item 7>

<item 8>
Florence Welch
Florence Leontine Mary Welch (born 28 August 1986) is an English musician, singer and songwriter, best known as the vocalist of indie rock band Florence and the
Machine.
</item 8>

<item 9>
Hayley Williams
Hayley Nichole Williams (born December 27, 1988) is an American singer, songwriter and musician.

 She serves as the lead vocalist, primary songwriter and occasional keyboardist of the rock band Paramore.

 The band was formed in 2004 by Josh Farro, Zac Farro, Jeremy Davis and Williams.

 The band consists of Hayley Williams, Zac Farro and Taylor York.

 The band has five studio albums: "All We Know Is Falling" (2005), "Riot!

" (2007), "Brand New Eyes" (2009), "Paramore" (2013) and "After Laughter" (2017).
</item 9>

<item 10>
Terry Ray (actor)
Terry Ray (born February 12, 1961) is an American actor, screenwriter, and producer.

 Some of his work includes "Gaydar" and "Cost of Living" (2009).

 Ray is the creator and writer of "here!

 TV"'s sitcom "From Here on OUT", the first original gay sitcom created by a gay network.

 Terry stars in the sitcom, along with Juliet Mills, Suzanne Whang and T.J. Hoban.

<br><br>Terry Ray's 1987 appearance on Scrabble (game show) is noted as number 3 in the "10 Great Gay Moments in Game Show History", after Charles Nelson Reilly
and Paul Lynde.

 In the same appearance, Terry was also named as one of the "5 Weirdest Game Show Contestants Ever", by Movieline.

<br><br>Terry’s first professional writing job was the stage performance “Hollywood Goes Classical" at the venerable Dorothy Chandler Pavilion starring Mickey
Rooney, Michael York, Dean Jones, Rhonda Fleming and several other classic film stars.
</item 10>
=== END ITEM id=2963bf831b452787 ===

=== ITEM id=5a86d3988cd416b7 ===
Gold sentences and facts:
Sentence 0:  The film stars Juno Temple, Julia Garner, Alessandro Nivola, Maggie Siff, Olivia Luccardi and Philip Ettinger.
  Fact 0.0: The film stars Juno Temple.
  Fact 0.1: The film stars Julia Garner.
  Fact 0.2: The film stars Alessandro Nivola.
  Fact 0.3: The film stars Maggie Siff.
  Fact 0.4: The film stars Olivia Luccardi.
  Fact 0.5: The film stars Philip Ettinger.
Sentence 1:  She has appeared in films such as " Black Mass", "The Other Boleyn Girl", "Wild Child", "Atonement", "Maleficent", "The Three Musketeers" and "The Dark Knight Rises".
  Fact 1.0: The actress has appeared in the film Black Mass.
  Fact 1.1: The actress has appeared in the film The Other Boleyn Girl.
  Fact 1.2: The actress has appeared in the film Wild Child.
  Fact 1.3: The actress has appeared in the film Atonement.
  Fact 1.4: The actress has appeared in the film Maleficent.
  Fact 1.5: The actress has appeared in the film The Three Musketeers.
  Fact 1.6: The actress has appeared in the film The Dark Knight Rises.

Retrieved items:
<item 1>
Jack &amp; Diane (film)
Jack & Diane is a 2012 American romantic horror film written and directed by Bradley Rust Gray, and starring Riley Keough and Juno Temple.

 Olivia Thirlby and Ellen Page were originally cast for the lead roles in 2008 before the project was postponed.
</item 1>

<item 2>
Juliet Weber
Juliet Weber is a producer, director, writer, and an editor for the film industry.

 She co-produced films such as Fastpitch (2000) and Unchained Memories (2003).
</item 2>

<item 3>
Magic Magic (2013 film)
Magic Magic is a 2013 American-Chilean psychological thriller film written and directed by Sebastián Silva and starring Juno Temple, Emily Browning, Michael
Cera, and Catalina Sandino Moreno.
</item 3>

<item 4>
Juno Temple
Juno Temple (born 21 July 1989) is an English actress.

 She has appeared in films such as " Black Mass", "The Other Boleyn Girl", "Wild Child", "Atonement", "Maleficent", "The Three Musketeers" and "The Dark Knight
Rises".

 She also had a starring role in the HBO period drama series "Vinyl".
</item 4>

<item 5>
Lovelace (film)
Lovelace is a 2013 American biographical drama film about porn actress Linda Boreman, better known as Linda Lovelace, star of "Deep Throat", a seminal 1972 film
at the forefront of the Golden Age of Porn.

 "Lovelace" covers her life from age 20 to 32.

 Directed by Rob Epstein and Jeffrey Friedman, the film was written by Andy Bellin and stars Amanda Seyfried, Peter Sarsgaard, Sharon Stone, Adam Brody, James
Franco, Chloë Sevigny, and Juno Temple.

 The film had its world premiere on January 22, 2013, at the 2013 Sundance Film Festival and opened in a U.S. limited release on August 9, 2013.
</item 5>

<item 6>
Deadfall (2012 film)
Deadfall is a 2012 American crime drama film directed by Stefan Ruzowitzky, written by Zach Dean and starring Eric Bana, Olivia Wilde, and Charlie Hunnam.
</item 6>

<item 7>
Kate Winslet filmography
Kate Winslet is a British actress and singer who has appeared in numerous films and television series.

 Her film debut was as Juliet Hulme in the 1994 film "Heavenly Creatures".

 She went on to appear in the later 1990s films "Sense and Sensibility" as Marianne Dashwood (1995), which earned her a nomination for the Academy Award for
Best Supporting Actress, "Jude" as Sue Bridehead (1996), "Hamlet" as Ophelia (1996) and "Titanic" as Rose DeWitt Bukater (1997).

 For her role in "Titanic", it earned her Best Actress nominations for an Academy Award, Golden Globe and Screen Actors Guild Award.
</item 7>

<item 8>
Dirty Girl (2010 film)
Dirty Girl is a 2010 American coming of age comedy-drama film written and directed by Abe Sylvia.

 It stars Juno Temple, Milla Jovovich and William H. Macy.

 It premiered at the 2010 Toronto International Film Festival on September 12, 2010.

 It was distributed theatrically by The Weinstein Company on October 7, 2011.
</item 8>

<item 9>
Cracks (film)
Cracks is a 2009 independent drama psychological thriller film starring Eva Green, James McGovern, Juno Temple, María Valverde, and Imogen Poots, which was
released theatrically in the UK and Ireland on 4 December 2009.

 In the United States it was released by IFC Films on 18 March 2011, and premiered on Showtime in late 2011.
</item 9>

<item 10>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 10>
=== END ITEM id=5a86d3988cd416b7 ===

=== ITEM id=4fc92a6f338092ba ===
Gold sentences and facts:
Sentence 0: August or Auguste Brancart (21 July 1851 - 1894?)
  Fact 0.0: August Brancart is also known as Auguste Brancart.
  Fact 0.1: August Brancart was born on 21 July 1851.
  Fact 0.2: The end date given for August Brancart is 1894, with a question mark.
Sentence 1:  He also published work of the Decadent movement such as "Monsieur Vénus" by Rachilde.
  Fact 1.0: Brancart published work of the Decadent movement.
  Fact 1.1: Brancart published "Monsieur Vénus" by Rachilde.
  Fact 1.2: "Monsieur Vénus" by Rachilde is an example of Decadent movement work.
Sentence 2: The Decadent Movement was a late 19th-century artistic and literary movement, centered in Western Europe, that followed an aesthetic ideology of excess and artificiality.
  Fact 2.0: The Decadent Movement was an artistic and literary movement.
  Fact 2.1: The Decadent Movement was a late 19th-century movement.
  Fact 2.2: The Decadent Movement was centered in Western Europe.
  Fact 2.3: The Decadent Movement followed an aesthetic ideology of excess.
  Fact 2.4: The Decadent Movement followed an aesthetic ideology of artificiality.

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
Albert Camus
Albert Camus (] ; 7 November 1913 – 4 January 1960) was a French philosopher, author, and journalist.

 His views contributed to the rise of the philosophy known as absurdism.

 He wrote in his essay "The Rebel" that his whole life was devoted to opposing the philosophy of nihilism while still delving deeply into individual freedom.

 He won the Nobel Prize in Literature in 1957.
</item 2>

<item 3>
Christian de la Mazière
Christian de la Mazière (August 22, 1922 in Tunis – February 15, 2006) was a journalist and member of the Charlemagne Division of the Waffen SS.

 He is known for discussing his role in the documentary "The Sorrow and the Pity" and also wrote a book titled "The Captive Dreamer".

 At the start of the war, he served in the French Army and was later retained in the military of Vichy France until 1942.

 After being discharged, he worked for the fascist newspaper "Le Pays Libre", joining the Charlemagne Division just before the Liberation of Paris in 1944.

 He was taken prisoner in Pomerania by Polish forces in the Red Army.
</item 3>

<item 4>
Auguste Brancart
August or Auguste Brancart (21 July 1851 - 1894?)

 was a Belgian publisher of pornographic literature, credited with the first publication of "My Secret Life".

 He published translations of English pornography into French and vice versa for English publishers such as Edward Avery.

 He also published work of the Decadent movement such as "Monsieur Vénus" by Rachilde.
</item 4>

<item 5>
The Decadent Movement, originating in late-19th-century France, prized excess, artificiality, perversion, skepticism, and creativity over nature and logic; key
examples include Rops and Huysmans’s *Against Nature*. In 1959 Camus adapted Dostoyevsky’s *The Possessed*, directing an acclaimed, technically ambitious
four-hour production. Ionesco, a Romanian-French absurdist playwright, explored human insignificance in works including *The Lesson*, *
</item 5>

<item 6>
The entries survey literature, theatre, film, music, and curiosities: French Decadence favored artifice and excess; Camus staged Dostoyevsky’s *The Possessed*
in 1959; Algren won the 1950 National Book Award; Poe satirized Gothic fiction; Moorcock wrote Elric and science-fiction novels; Nin’s diaries and erotica
appeared partly posthumously. Wilde’s *Dorian Gray* faced censorship and condemnation. Verne’s
</item 6>

<item 7>
Marcel J. Melançon
Marcel J. Mélançon is a Canadian philosopher and scientist.

 He was born in Saint-Barnabé, Quebec, in 1938, and is mostly known for his book on the French philosopher Albert Camus "Albert Camus, An Analysis of his
thought" (Albert Camus.

 Analyse de sa pensée).

 He has mostly worked together with the biologist Richard D. Lambert.
</item 7>

<item 8>
The entries cover pornographic publisher Auguste Brancart; Albert Camus, his absurdist philosophy, Nobel Prize, fiction and essays opposing nihilism and capital
punishment; and his biographers. Other topics include the disputed Smiley Face murder theory; Joe McGinniss, Jeffrey MacDonald and the Fatal Vision controversy;
the Guantánamo memoir Enemy Combatant; New Zealand’s Parker–Hulme case and Mazengarb Report; the Lindbergh kidnapping; Nazi collaborator Christian de
</item 8>

<item 9>
The Rebel (book)
The Rebel (French: "L'Homme révolté" ) is a 1951 book-length essay by Albert Camus, which treats both the metaphysical and the historical development of
rebellion and revolution in societies, especially Western Europe.

 Camus relates writers and artists as diverse as Epicurus and Lucretius, Marquis de Sade, Georg Wilhelm Friedrich Hegel, Fyodor Dostoyevsky, Friedrich
Nietzsche, Max Stirner, André Breton, and others in an integrated, historical portrait of man in revolt.

 Examining both rebellion and revolt, which may be seen as the same phenomenon in personal and social frames, Camus examines several 'countercultural' figures
and movements from the history of Western thought and art, noting the importance of each in the overall development of revolutionary thought and philosophy.
</item 9>

<item 10>
Decadent movement
The Decadent Movement was a late 19th-century artistic and literary movement, centered in Western Europe, that followed an aesthetic ideology of excess and
artificiality.

 The visual artist Félicien Rops's body of work and Joris-Karl Huysmans's novel "Against Nature" (1884) are considered the prime examples of the decadent
movement.

 It first flourished in France and then spread throughout Europe and to the United States.

 The movement was characterized by self-disgust, sickness at the world, general skepticism, delight in perversion and employment of crude humor and a belief in
the superiority of human creativity over logic and the natural world.
</item 10>
=== END ITEM id=4fc92a6f338092ba ===

=== ITEM id=4f10c78b230c4ade ===
Gold sentences and facts:
Sentence 0: Keisuke Kuwata (桑田 佳祐 , Kuwata Keisuke , born February 26, 1956) is a Japanese multi-instrumentalist, singer-songwriter, and frontman for the Southern All Stars, as well of his own solo band, the Kuwata band.
  Fact 0.0: Keisuke Kuwata's name is written 桑田 佳祐 (Kuwata Keisuke).
  Fact 0.1: Keisuke Kuwata was born February 26, 1956.
  Fact 0.2: Keisuke Kuwata is a Japanese multi-instrumentalist.
  Fact 0.3: Keisuke Kuwata is a Japanese singer-songwriter.
  Fact 0.4: Keisuke Kuwata is frontman for the Southern All Stars.
  Fact 0.5: Keisuke Kuwata has his own solo band, the Kuwata band.
Sentence 1: Moya Brennan, born Máire Ní Bhraonáin (] ), also known as Máire Brennan (born 4 August 1952), is an Irish folk singer, songwriter, harpist, and philanthropist.
  Fact 1.0: Moya Brennan was born Máire Ní Bhraonáin.
  Fact 1.1: Moya Brennan is also known as Máire Brennan.
  Fact 1.2: Moya Brennan was born 4 August 1952.
  Fact 1.3: Moya Brennan is an Irish folk singer.
  Fact 1.4: Moya Brennan is an Irish songwriter.
  Fact 1.5: Moya Brennan is an Irish harpist.
  Fact 1.6: Moya Brennan is an Irish philanthropist.

Retrieved items:
<item 1>
Moya Brennan
Moya Brennan, born Máire Ní Bhraonáin (] ), also known as Máire Brennan (born 4 August 1952), is an Irish folk singer, songwriter, harpist, and philanthropist.

 She is the older sister of Enya and Brídín Brennan.

 She began performing professionally in 1970 when her family formed the band Clannad, and is considered as the "First Lady of Celtic Music".

 Moya released her first solo album in 1992 called "Máire", a successful venture.

 She has been nominated for two Grammys and has won an Emmy Award.

 She has recorded music for several soundtracks, including "Titanic", "To End All Wars" and "King Arthur".
</item 1>

<item 2>
Keiko Utoku
Keiko Utoku (宇徳敬子 , "Utoku Keiko" , born April 7, 1967 in Izumi, Kagoshima, Japan) is a Japanese singer and songwriter under UK Sweet label.Keiko made first
debut with chorus part in band B.B.Queens then joined with J-pop female band Mi-Ke from 1991 to 1993, before starting a solo career.

 As a child, Utoku listened to a lot of music, including music made by Momoe Yamaguchi.

 She has written and sung many songs, such as "Wasurenagusa", and Sonic the Hedgehog fans remember her for her vocal work in the original version of the Sonic
CD video game's opening and closing themes in Japan and Europe, "Sonic – You Can Do Anything" and "Cosmic Eternity – Believe In Yourself" in 1993.

 In addition, she also provided Sonic's vocals in the game.

 At recent dates, she has released 15 singles, 5 digital singles, 4 studio albums, 1 mini and 2 best albums.
</item 2>

<item 3>
Óró – A Live Session
Óró – A Live Session is a music album by Irish musician Máire Brennan, now known as "Moya Brennan".

 Just before the start of the German tour, Moya and her band recorded a "live session" album to have available for fans on the tour.

 She's tried to recreate some of the live feel with the band in the studio covering many of the songs she plays (and was about to play) in concert.

 The album will become more widely available subsequently, but originally was released in 2005.
</item 3>

<item 4>
Two Horizons
Two Horizons is the fifth solo album by Irish singer Moya Brennan.

 It is her first full-length release under the name "Moya Brennan", as opposed to "Máire Brennan".

 The album was predominantly recorded in her home studio in Dublin and was nominated for a Grammy award.

 The album was recorded between 2002–2003 and first became available on 23 October 2003.

 It is also Brennan's first non-Christian album since her 1993 recording Misty Eyed Adventures.

 It is the most successful of her solo albums to date and the second to be Grammy-nominated.
</item 4>

<item 5>
Kevin Mayer
Kevin Mayer (] , ] or ] , born 10 February 1992) is a French athlete specializing in combined events.

 A native of Drôme in southeast France, Mayer excelled in combined events from a young age; highlights from his youth career include winning gold medals in the
octathlon at the 2009 World Youth Championships and in the decathlon at the 2010 World Junior Championships and at the 2011 European Athletics Junior
Championships, establishing a new national junior record at the latter.
</item 5>

<item 6>
2014 Monterrey Open – Doubles
Tímea Babos and Kimiko Date-Krumm were the defending champions, but chose not to participate together.

 Babos played alongside Olga Govortsova, while Date-Krumm teamed up with Karolína Plíšková.

 The two teams were scheduled to meet in the semifinals, but Date-Krumm withdrew with a right leg injury.

<br>
</item 6>

<item 7>
Keisuke Kuwata
Keisuke Kuwata (桑田 佳祐 , Kuwata Keisuke , born February 26, 1956) is a Japanese multi-instrumentalist, singer-songwriter, and frontman for the Southern All
Stars, as well of his own solo band, the Kuwata band.

 He has also done significant amount of scoring music for films.

 He went to Aoyama Gakuin University.
</item 7>

<item 8>
My Match Is a Makin'
My Match Is A Makin' (English translation of opening track title) is a music album by Irish musicians Moya Brennan and Cormac de Barra.

 This is Moya's eighth studio album to be released.

 It was released on 14 April 2010 exclusively to concert goers on her Spring 2010 tour of the Netherlands.

 The album was re-released to a wider audience under the title Voices & Harps.
</item 8>

<item 9>
Bruce Yonemoto
Bruce Yonemoto (born 1949) is a Japanese-American multimedia artist.

 His work has been recognized by the National Endowment for the Arts, the American Film Institute, The Rockefeller Foundation, the Maya Deren Award for
Experimental Film and Video, and a mid-career survey show at the Japanese American National Museum, in addition to major solo exhibitions at the
InterCommunication Center in Tokyo, the Institute of Contemporary Art the University of Pennsylvania, and the Kemper Museum in Kansas City, MO.

 Yonemoto was featured in the 2002 Corcoran Gallery Bienniel, in Washington D.C. Yonemoto was also a recipient of the Creative Capital Visual Arts Award in
2008.
</item 9>

<item 10>
Kate Berlant
Kate Berlant (born July 16, 1987) is an American stand-up comedian, actress, and writer.

 She is known for appearing on "The Meltdown with Jonah and Kumail" and starring in an episode of the Netflix original series "".

 With John Early, Berlant created the Vimeo original series "555" produced by Abso Lutely Productions.

 Her father is the artist Tony Berlant.
</item 10>
=== END ITEM id=4f10c78b230c4ade ===
