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

=== ITEM id=f4c14ff1e4f7de5b ===
Gold sentences and facts:
Sentence 0:  Պողոսյան , born July 15, 1990) is an American professional ice hockey defenseman currently playing for the Buffalo Sabres.
  Fact 0.0: Պողոսյան was born on July 15, 1990.
  Fact 0.1: Պողոսյան is an American professional ice hockey defenseman.
  Fact 0.2: Պողոսյան is currently playing for the Buffalo Sabres.
Sentence 1: Daron Vartan Malakian (Armenian: Տարօն/Տարոն Վարդան Մալաքեան, born July 18, 1975) is an Armenian–American singer-songwriter, multi-instrumentalist, and record producer.
  Fact 1.0: The Armenian name of Daron Vartan Malakian is Տարօն/Տարոն Վարդան Մալաքեան.
  Fact 1.1: Daron Vartan Malakian was born on July 18, 1975.
  Fact 1.2: Daron Vartan Malakian is an Armenian-American singer-songwriter.
  Fact 1.3: Daron Vartan Malakian is a multi-instrumentalist.
  Fact 1.4: Daron Vartan Malakian is a record producer.

Retrieved items:
<item 1>
Zach Bogosian
Zachary M. Bogosian (Armenian: Զաքարի Մ.

 Պողոսյան , born July 15, 1990) is an American professional ice hockey defenseman currently playing for the Buffalo Sabres.

 Bogosian attended Cushing Academy in Massachusetts before he joined the Peterborough Petes of the Ontario Hockey League (OHL) in 2006.

 He played two seasons in the OHL, and was nominated for the Red Tilson Trophy as the league's most outstanding player in his second season.

 Bogosian was regarded as a complete, physical defenseman who could contribute on both offense and defense; he was rated as one of the top players heading into
the 2008 NHL Entry Draft, where the Atlanta Thrashers selected him third overall.

 He signed a contract with the team a few weeks after the draft and began the season with the Thrashers, though he missed several weeks of his rookie year due
to injury.

 In his second season, he tied a team record for goals by a defenseman.
</item 1>

<item 2>
Daron Jones
Daron Tavaris Jones (born December 27, 1976) is an American singer-songwriter, record producer.

 He is a member in Bad Boy Records group 112.

 His debut solo album titled "Uncensored" was released December 5, 2010.

 The first single from the album is the digitally-released "Money."

 Daron digitally released "Christmas in Atlanta" in December 2014, which reached the top 30 on the iTunes R&B chart.

 Daron is currently in the studio working on new music.
</item 2>

<item 3>
Greg Davis (musician)
Greg Davis is an American electronic musician who has recorded albums drawing from a wide variety of sources, including guitar, field recording, various
world/ethnic/traditional instruments, percussion, and voice, all processed through digital manipulation.

 After years of experience in hip-hop groups, jazz combos, free improvisation, and experimental composition, Davis moved toward computer-based music in 1997.

 At Depaul University in Chicago, he studied classical and jazz guitar alongside composition and jazz studies.

 In 1997, he started his own label, Autumn Records, to release his own music and the music of others.

 Davis moved to Boston in 1999 to pursue a master's degree in composition (which he received in June 2001) while independently immersing himself in the study of
electronic music.

 Since 2001, he has played shows all around the world and released records on a number of different labels.

 He has collaborated with Keith Fullerton Whitman, Ben Vida, Sebastien Roux, Chris Weisman, Steven Hess, Jeph Jerman, Akron/Family, Toby Aronson (as
Harmonizer), and Zach Wallace (as Sun Circle).
</item 3>

<item 4>
Jimmy Chamberlin
James Joseph Chamberlin (born June 10, 1964) is an American drummer and record producer.

 He is best known as the drummer for the alternative rock band The Smashing Pumpkins.

 Following the 2000 breakup of the band, Chamberlin joined Pumpkins frontman Billy Corgan in the supergroup Zwan and also formed his own group, the Jimmy
Chamberlin Complex.

 In 2005, Chamberlin joined Corgan in reforming The Smashing Pumpkins; he eventually left the group in March 2009, though he returned again in 2015 for a summer
tour.

 He performed in the group Skysaw until 2012.

 He is currently active under the Jimmy Chamberlin Complex name.

 In addition to his current work as CEO, Chamberlin has joined Chicago jazz saxophonist Frank Catalano for a string of 2013–15 performances in the Chicago area.

 An EP by Catalano and Chamberlin "Love Supreme Collective - EP" was released on the 29th of July 2014.
</item 4>

<item 5>
Zach Bogosian
In his second season, he tied a team record for goals by a defenseman.

 Bogosian first played in an international tournament when he joined the American national team at the 2009 IIHF World Championship.
</item 5>

<item 6>
Daron Malakian
Daron Vartan Malakian (Armenian: Տարօն/Տարոն Վարդան Մալաքեան, born July 18, 1975) is an Armenian–American singer-songwriter, multi-instrumentalist, and record
producer.

 He is best known as the guitarist, songwriter and second vocalist of the rock band System of a Down and as the lead vocalist, lead guitarist and songwriter of
the band Scars on Broadway.

 Like the rest of the Hollywood-based band System of a Down, he is of Armenian ancestry, but he is the only member to have been born in the United States.

 Daron Malakian is known for his distinctive playing and is ranked 40th in Loudwire's list of "Top 50 Hard Rock + Metal Guitarists Of All Time" and #11 in
MusicRadar's poll, "The 20 Greatest Metal Guitarists Ever".

 He is placed 30th in Guitar World's List of The 100 Greatest Heavy Metal Guitarists of All Time.
</item 6>

<item 7>
Bad Acid Trip
Bad Acid Trip is an American metal band, currently signed to System of a Down lead singer Serj Tankian's record label, Serjical Strike Records.

 Their 2004 album, "Lynch the Weirdo", was produced by Daron Malakian, another System of a Down member.

 In 2006 they performed on the second stage at the Ozzfest summer tour.

 The band has three full-length releases ("For the Weird by the Weird", "Lynch the Weirdo", and "Humanly Possible") and multiple EPs.
</item 7>

<item 8>
Lazzo (musician)
Lazzo is a producer, engineer, and songwriter specializing in the rock, electronic, dance, and pop genres.

 In late 2013, he released his Wammy (Washington Area Music Association / WAMA) award-winning dance/dubstep remix album with Rites of Ash, Kept Me Up All Night.

 In Oct. 2014, Rites of Ash released their new full-length album titled "Kill For Love".

 He has had his music featured on numerous MTV shows, FUSE, MTV2, and MTVu, as well as shared the stage with many national artists.
</item 8>

<item 9>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 9>

<item 10>
Profiles span musicians across jazz, rock, metal, electronic, gospel, and pop. Robert Glasper has three Grammys; Carlos Santana, ten Grammys and a landmark
Latin-rock fusion career. Daron Malakian (System of a Down), Gaahl (Gorgoroth), Roy Khan (Kamelot), Toshi (X Japan), and Rivermaya’s former guitarist Perfecto
de Castro are noted rock figures. Relient K’s Matt Thiessen, electronic artists Greg Davis and
</item 10>
=== END ITEM id=f4c14ff1e4f7de5b ===

=== ITEM id=13dbeb5038cd56ff ===
Gold sentences and facts:
Sentence 0: Migene González-Wippler is a Puerto Rican new-age author and a leading expert on the Afro-Caribbean religion of Santería.
  Fact 0.0: Migene González-Wippler is a Puerto Rican.
  Fact 0.1: Migene González-Wippler is a new-age author.
  Fact 0.2: Migene González-Wippler is a leading expert on the Afro-Caribbean religion of Santería.
Sentence 1: Santería, also known as Regla de Ochá, La Regla de Ifá, or Lucumi, is an Afro-American religion of Caribbean origin that developed in the Spanish Empire among West African descendants.
  Fact 1.0: Santería is also known as Regla de Ochá.
  Fact 1.1: Santería is also known as La Regla de Ifá.
  Fact 1.2: Santería is also known as Lucumi.
  Fact 1.3: Santería is an Afro-American religion.
  Fact 1.4: Santería is a religion of Caribbean origin.
  Fact 1.5: Santería developed in the Spanish Empire.
  Fact 1.6: Santería developed among West African descendants.

Retrieved items:
<item 1>
Ewe people
The Ewe people (Ewe: "Eʋeawó" , lit.

 "Ewe people"; or " Eʋedukɔ́ ", lit.

 "Ewe nation","Eʋenyigba" Eweland;) are an African ethnic group.

 They are the largest ethnic group in Togo (32%), the third largest ethnic group in Ghana (14%), and are a minority ethnic group in southern Benin, Ivory Coast
and Nigeria.

 They speak the Ewe language (Ewe: "Eʋegbe" ) which belongs to the Niger-Congo family of languages.

 They are related to other speakers of Gbe languages, such as, the Fon, Gen, Phla Phera, and the Aja people of Togo and Benin.
</item 1>

<item 2>
Guanche mummies
Guanche mummies are the intentionally desiccated remains of members of the indigenous Guanche people of the Canary Islands.

 The majority of Guanche mummies were made during the eras prior to Spanish settlement of the area in the 15th century.

 The methods of embalming are similar to those that were used by the Ancient Egyptians, though fewer mummies remain from the Guanche due to looting and
desecration.
</item 2>

<item 3>
Santería
Santería, also known as Regla de Ochá, La Regla de Ifá, or Lucumi, is an Afro-American religion of Caribbean origin that developed in the Spanish Empire among
West African descendants.

 "Santeria" is a Spanish word that means the "worship of saints".

 Santería is influenced by and syncretized with Roman Catholicism.

 Its sacred language, a variety of Yoruba, is the Lucumí language.
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
Mazahua language
The Mazahua language is an indigenous language of Mexico, spoken in the country's central states by the ethnic group that is widely known as the Mazahua but
calls itself the Hñatho.

 It is a Mesoamerican language and has many of the traits of the Mesoamerican Linguistic Area.

 In 2003, along with some 62 other indigenous languages, it was recognised by a statutory law of Mexico (General Law of Linguistic Rights of the Indigenous
Peoples) as an official language in the Federal District and the other administrative divisions in which it is spoken, and on an equal footing with Spanish.

 The largest concentration of Mazahua is found in the municipality of San Felipe del Progreso, State of México, near Toluca.
</item 5>

<item 6>
Maléku language
The Maléku Jaíka language, also called Guatuso, Watuso-Wétar, and Guetar, is an indigenous language of north central Costa Rica.

 It is a Chibchan language and Votic language spoken by around 300 to 750 indigenous Maléku people.

 This language is considered to be endangered according to The Endangered Languages Project.

 Corobicí is possibly a dialect.
</item 6>

<item 7>
Chiguata District
Chiguata is a district located in the Arequipa province, 30 km away from the city of Arequipa, Peru, on the slopes of the volcano Pikchu Pikchu (where at its
summit pre-Inca worshipped remains and artifacts have been found).

 The name "chiguata" comes from the Quechua words "chiri" meaning "cold" and "wata" meaning "year" (cold all year).

 Ancient Chiguata was inhabited by the local natives until the arrival of Spanish influence in charge of the emissary Diego Hernández de Córdova in 1540.

 Soon after, the construction of a small plaza was initiated, and according to the Dominican religious order and local sources, the plaza's "Templo del Espíritu
Santo" ("Sacred Spirit temple") along with its domes and lateral cover was not begun until 1739.

 The design of the temple is quite interesting, with two angels in feathered nightwear and their arms lifted vertically, separated by twelve fringes of
rectangular flowers which add to the whole decoration of the temple.
</item 7>

<item 8>
Ethnic religion
In religious studies, ethnic religion (or indigenous religion) is the expression of religion associated with a particular ethnic group.

 Ethnic religions are often distinguished from religions which claim to not be limited in ethnic or national scope, such as Christianity or Islam.

 Ethnic religions do not have to be excluded to independent religions.

 Some localized denominations of global religions are practiced solely by certain ethnic groups, with ethnic groups like the Assyrians following a unique
denominational structure of Christianity known as the Assyrian Church of the East.
</item 8>

<item 9>
Migene González-Wippler
Migene González-Wippler is a Puerto Rican new-age author and a leading expert on the Afro-Caribbean religion of Santería.

 González-Wippler was born in Puerto Rico and has degrees in psychology and anthropology from the University of Puerto Rico and from Columbia University.

 In addition to her solid background in social sciences she has also worked as a science editor for the Interscience Division of John Wiley, the American
Institute of Physics, and the American Museum of Natural History, and as an English editor for the United Nations in Vienna, where she resided for many years.

 She is a cultural anthropologist and lectures frequently at universities and other educational institutions.

 She also has contributed extensively to the Institute of Puerto Rican Culture's collection, especially when it comes to Santeria and its practices, beliefs and
organization.
</item 9>

<item 10>
Church of the Guanche People
The Church of the Guanche People (Spanish: "Iglesia del Pueblo Guanche" ) is a religious organisation, founded in 2001 in the city of San Cristóbal de La Laguna
(Tenerife, Canary Islands, Spain).

 It aims to perpetuate and spread the pagan religion of the ancient Guanche people.
</item 10>
=== END ITEM id=13dbeb5038cd56ff ===

=== ITEM id=ed0d2d8d098aae28 ===
Gold sentences and facts:
Sentence 0: More Hot Rocks (Big Hits & Fazed Cookies) is the second compilation album of Rolling Stones music released by former manager Allen Klein's ABKCO Records (who usurped control of the band's Decca/London material in 1970) after the band's departure from Decca and Klein.
  Fact 0.0: More Hot Rocks (Big Hits & Fazed Cookies) is the second compilation album of Rolling Stones music.
  Fact 0.1: More Hot Rocks was released by ABKCO Records.
  Fact 0.2: ABKCO Records is run by former manager Allen Klein.
  Fact 0.3: Allen Klein usurped control of the Rolling Stones' Decca/London material in 1970.
  Fact 0.4: More Hot Rocks was released after the Rolling Stones departed from Decca.
  Fact 0.5: More Hot Rocks was released after the Rolling Stones departed from Allen Klein.
Sentence 1: Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording artists.
  Fact 1.0: Allen Klein was born December 18, 1931.
  Fact 1.1: Allen Klein died July 4, 2009.
  Fact 1.2: Allen Klein was an American businessman.
  Fact 1.3: Allen Klein was a music publisher.
  Fact 1.4: Allen Klein was a writers' representative.
  Fact 1.5: Allen Klein was a filmmaker.
  Fact 1.6: Allen Klein was a record label executive.
  Fact 1.7: Allen Klein was most noted for his tough persona.
  Fact 1.8: Allen Klein was most noted for aggressive, innovative negotiation tactics.
  Fact 1.9: Many of Allen Klein's negotiation tactics established higher industry standards for compensating recording artists.
Sentence 2:  He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming one of the most powerful individuals in the music industry during his era.
  Fact 2.0: Allen Klein first scored massive monetary windfalls for Buddy Knox.
  Fact 2.1: Allen Klein first scored massive contractual windfalls for Buddy Knox.
  Fact 2.2: Allen Klein first scored massive monetary windfalls for Jimmy Bowen.
  Fact 2.3: Allen Klein first scored massive contractual windfalls for Jimmy Bowen.
  Fact 2.4: Buddy Knox and Jimmy Bowen were one-hit rockabillies of the late 1950s.
  Fact 2.5: Allen Klein later parlayed his early successes into a position managing Sam Cooke.
  Fact 2.6: Allen Klein eventually managed the Beatles.
  Fact 2.7: Allen Klein eventually managed the Rolling Stones.
  Fact 2.8: Allen Klein managed the Beatles and the Rolling Stones simultaneously.
  Fact 2.9: Allen Klein managed many other artists.
  Fact 2.10: Allen Klein became one of the most powerful individuals in the music industry during his era.

Retrieved items:
<item 1>
George Thorogood
George Thorogood (born February 24, 1950) is an American musician, singer and songwriter from Wilmington, Delaware.

 His "high-energy boogie-blues" sound became a staple of 1980s rock radio, with hits like his original songs "Bad to the Bone" and "I Drink Alone".

 He has also helped popularize older songs by American icons, such as "Move It on Over", "Who Do You Love?

", and "House Rent Boogie/One Bourbon, One Scotch, One Beer", which became staples of classic rock radio.
</item 1>

<item 2>
Bigtop Records
Bigtop Records was an American record label started by music executive Johnny Bienstock and the major music publisher Hill & Range Music and was co-owned along
with Big Top Record Distributors (sic).

 Hit artists included Del Shannon, Johnny and the Hurricanes, Lou Johnson, Sammy Turner, Don and Juan, and Toni Fisher.

 Big Top Record Distributors also distributed Paul Case's Dunes Records label in the early 60's, which had hits from Ray Peterson ("Corrina, Corrina") and
Curtis Lee ("Pretty Little Angel Eyes"), both records produced by Phil Spector.

 Bell Records briefly distributed Bigtop prior to the label closing, around 1965.

 "Bigtop" also released two Mad Magazine music parody themed albums; "Mad Twists Rock 'N' Roll" and "Fink Along With Mad" in 1963.
</item 2>

<item 3>
Howie Klein
Howie Klein (born 1950) is an American writer, concert promoter, disc jockey, music producer, record label founder, record label executive, progressive
political activist, and adjunct professor of music.

 He is perhaps best known for his role as President of Reprise Records from 1989 to 2001.

 He appears occasionally as himself in music-related film documentaries and has received accolades for his stance against censorship and for his advocacy of
free speech protection.
</item 3>

<item 4>
More Hot Rocks (Big Hits &amp; Fazed Cookies)
More Hot Rocks (Big Hits & Fazed Cookies) is the second compilation album of Rolling Stones music released by former manager Allen Klein's ABKCO Records (who
usurped control of the band's Decca/London material in 1970) after the band's departure from Decca and Klein.

 Released in late 1972 as the follow-up to the hugely successful "Hot Rocks 1964–1971", it was another success for Klein and ABKCO.
</item 4>

<item 5>
Hot Rocks 1964–1971
Hot Rocks 1964–1971 is the first compilation album of Rolling Stones music released by former manager Allen Klein's ABKCO Records (who gained control of the
band's Decca/London material in 1970) after the band's departure from Decca and Klein.

 Released in late 1971, it proved to be The Rolling Stones' biggest-selling release of their career and an enduring and popular retrospective.
</item 5>

<item 6>
Hot Streets
Hot Streets is the tenth studio album (twelfth overall) by the American band Chicago, released in 1978.

 In many ways, "Hot Streets" marked the beginning of a new era for the band, turning to disco music, a move which would be derided in retrospect.

 It was also the band's first album with all-new material released since their second that didn't have a numbered title.

 It was also the first album not to feature original guitarist/vocalist Terry Kath, who died from an accidental self-inflicted gunshot wound in January 1978.

 He was replaced by Donnie Dacus on this album.
</item 6>

<item 7>
Allen Klein
Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most
noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording
artists.

 He founded ABKCO Music & Records Incorporated.

 Klein revolutionized the income potential of recording artists, who previously had been routinely victimized by onerous record company contracts.

 He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early
successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming
one of the most powerful individuals in the music industry during his era.
</item 7>

<item 8>
Music West Records
Music West Records was an independent record company founded by Allan Kaplan on December 1985 in San Rafael, California.

 The company was initially formed to promote Ray Lynch.

 During its run, artists released under the record company included Jim Chappell, Kenneth Nash, Chris Spheeris, and Øystein Sevåg.

 According to Gary Chappell, the manufacturer for Music West, the artists originated independently, claiming that the company's idea "has a statement that comes
directly from the
</item 8>

<item 9>
Sargent House
Sargent House is an American management company and record label based in Los Angeles.

 It was founded in June 2006 by Cathy Pellow, who is a music video commissioner for Atlantic Records and also owns a music video production company called
Refused TV.

 Pellow began managing the rock outfit RX Bandits in 2006.

 The band needed to release its new album "...And the Battle Begun", so Pellow decided to launch her own record label (also to be able to do things differently
from the traditional model of record labels).

 From that point onward, Pellow continued to manage bands and release records, all under the Sargent House banner.
</item 9>

<item 10>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 10>
=== END ITEM id=ed0d2d8d098aae28 ===

=== ITEM id=d02b44a1b37c37ff ===
Gold sentences and facts:
Sentence 0: Erving Goffman (11 June 1922 – 19 November 1982) was a Canadian-American sociologist and writer, considered by some "the most influential American sociologist of the twentieth century".
  Fact 0.0: Erving Goffman was born on 11 June 1922.
  Fact 0.1: Erving Goffman died on 19 November 1982.
  Fact 0.2: Erving Goffman was a Canadian-American sociologist.
  Fact 0.3: Erving Goffman was a writer.
  Fact 0.4: Some consider Erving Goffman the most influential American sociologist of the twentieth century.
Sentence 1:  In 2007 he was listed by "The Times Higher Education Guide" as the sixth most-cited author in the humanities and social sciences, behind Anthony Giddens, Pierre Bourdieu and Michel Foucault, and ahead of Jürgen Habermas.
  Fact 1.0: In 2007 Erving Goffman was listed by The Times Higher Education Guide.
  Fact 1.1: The Times Higher Education Guide listed Goffman as the sixth most-cited author in the humanities and social sciences in 2007.
  Fact 1.2: In that 2007 listing, Anthony Giddens, Pierre Bourdieu and Michel Foucault were ranked ahead of Goffman.
  Fact 1.3: In that 2007 listing, Goffman was ranked ahead of Jürgen Habermas.
Sentence 2: Paul-Michel Foucault (15 October 1926 – 25 June 1984), generally known as Michel Foucault (] ), was a French philosopher, historian of ideas, social theorist, and literary critic.
  Fact 2.0: Paul-Michel Foucault was born on 15 October 1926.
  Fact 2.1: Paul-Michel Foucault died on 25 June 1984.
  Fact 2.2: Paul-Michel Foucault is generally known as Michel Foucault.
  Fact 2.3: Michel Foucault was a French philosopher.
  Fact 2.4: Michel Foucault was a historian of ideas.
  Fact 2.5: Michel Foucault was a social theorist.
  Fact 2.6: Michel Foucault was a literary critic.

Retrieved items:
<item 1>
The Presentation of Self in Everyday Life
The Presentation of Self in Everyday Life is a 1956 sociology book by Erving Goffman, in which Goffman uses the imagery of the theatre in order to portray the
importance of human social interaction.

 Originally published in Scotland in 1956 and in the United States in 1959, it is Goffman’s first and most famous book, for which he received the American
Sociological Association’s MacIver award in 1961.
</item 1>

<item 2>
Ray Birdwhistell
Collaborations with others, including initially Margaret Mead and Gregory Bateson, and later, Erving Goffman and Dell Hymes had huge influence on his work.

 For example, the book he is best known for, "Kinesics and Context", "would not have appeared if it had not been envisaged by Erving Goffman" and he explicitly
stated "the paramount and sustaining influence upon my work has been that of anthropological linguistics", a tradition most directly represented at the
University of Pennsylvania by Hymes.
</item 2>

<item 3>
H. L. Mencken
Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.

 Known as the "Sage of Baltimore", he is regarded as one of the most influential American writers and prose stylists of the first half of the twentieth century.

 He commented widely on the social scene, literature, music, prominent politicians and contemporary movements.

 His satirical reporting on the Scopes trial, which he dubbed the "Monkey Trial", also gained him attention.
</item 3>

<item 4>
Dramaturgy (sociology)
Dramaturgy is a sociological perspective commonly used in microsociological accounts of social interaction in everyday life.

 The term was first adapted into sociology from the theatre by Erving Goffman, who developed most of the related terminology and ideas in his 1959 book, "The
Presentation of Self in Everyday Life".

 Kenneth Burke, whom Goffman would later acknowledge as an influence, had earlier presented his notions of dramatism in 1945, which in turn derives from
Shakespeare.

 However, the fundamental difference between Burke's and Goffman's view is that Burke believed that life was in fact theatre, whereas Goffman viewed theatre as
a metaphor.

 If we imagine ourselves as directors observing what goes on in the theatre of everyday life, we are doing what Goffman called dramaturgical analysis, the study
of social interaction in terms of theatrical performance.
</item 4>

<item 5>
George Orwell bibliography
The bibliography of George Orwell includes journalism, essays, novels and non-fiction books written by the British writer Eric Blair (1903–50), either under his
own name or, more usually, under his pen name George Orwell.

 Orwell was a prolific writer on topics related to contemporary English society and literary criticism, whom the British newsweekly "The Economist" in 2008
declared "perhaps the 20th century's best chronicler of English culture."

 His non-fiction cultural and political criticism constitutes the majority of his work, but Orwell also wrote in several genres of fictional literature.
</item 5>

<item 6>
Erving Goffman
Erving Goffman (11 June 1922 – 19 November 1982) was a Canadian-American sociologist and writer, considered by some "the most influential American sociologist
of the twentieth century".

 In 2007 he was listed by "The Times Higher Education Guide" as the sixth most-cited author in the humanities and social sciences, behind Anthony Giddens,
Pierre Bourdieu and Michel Foucault, and ahead of Jürgen Habermas.
</item 6>

<item 7>
Identity Performance
In everyday interactions, the body serves as a critical site of identity performance.

 In conveying who we are to other people, we use our bodies to project information about ourselves.

 This is done through movement, clothes, speech, and facial expressions.

 What we put forward is our best effort at what we want to say about who we are.

 Yet while we intend to convey one impression, our performance is not always interpreted as we might expect.

 Through learning to make sense of others’ responses to our behavior, we can assess how well we have conveyed what we intended.

 We can then alter our performance accordingly.

 This process of performance, interpretation, and adjustment is what Erving Goffman calls impression management.

 Impression management is a part of a larger process where people seek to define a situation through their behavior.

 People seek to define social situations by using contextual cues from the environment around them.

 Social norms emerge from situational definitions, as people learn to read cues from the environment and the people present to understand what is appropriate
behavior.
</item 7>

<item 8>
Goffman (disambiguation)
Erving Goffman (1922–1982) was a Canadian sociologist
</item 8>

<item 9>
Michel Foucault
Paul-Michel Foucault (15 October 1926 – 25 June 1984), generally known as Michel Foucault (] ), was a French philosopher, historian of ideas, social theorist,
and literary critic.
</item 9>

<item 10>
William H. Gass
William Howard Gass (born July 30, 1924) is an American novelist, short-story writer, essayist, critic, and former philosophy professor.

 He has written three novels, three collections of short stories, a collection of novellas, and seven volumes of essays, three of which have won National Book
Critics Circle Award prizes and one of which, "A Temple of Texts" (2006), won the Truman Capote Award for Literary Criticism.

 His 1995 novel "The Tunnel" received the American Book Award.

 His 2013 novel "Middle C" won the 2015 William Dean Howells Medal.
</item 10>
=== END ITEM id=d02b44a1b37c37ff ===

=== ITEM id=1e6a89e24fd1b4e5 ===
Gold sentences and facts:
Sentence 0: Haringey Heartlands is a currently ongoing urban regeneration project in the London Borough of Haringey, in north London, England.
  Fact 0.0: Haringey Heartlands is an urban regeneration project.
  Fact 0.1: Haringey Heartlands is currently ongoing.
  Fact 0.2: Haringey Heartlands is in the London Borough of Haringey.
  Fact 0.3: The London Borough of Haringey is in north London, England.
Sentence 1: The Mall Wood Green is a large shopping centre and residential complex in Wood Green, north London.
  Fact 1.0: The Mall Wood Green is a large shopping centre.
  Fact 1.1: The Mall Wood Green is a residential complex.
  Fact 1.2: The Mall Wood Green is in Wood Green, north London.

Retrieved items:
<item 1>
The Mall Wood Green
The Mall Wood Green is a large shopping centre and residential complex in Wood Green, north London.

 It is generally still referred to by its former name of Wood Green Shopping City; the signage on the building still uses that name.
</item 1>

<item 2>
Heartlands High School
Heartlands High School is a state secondary school located in Wood Green in the London Borough of Haringey.

 It welcomed its first students in September 2010.

 The headteacher is Mr. Simon Garrill.
</item 2>

<item 3>
London Skolars
London Skolars R.L.F.C. are a professional rugby league club based at the New River Stadium, Wood Green, Haringey in north London.

 They were founded in 1995 and have been professional since 2003, operating in the Kingstone Press League 1 in the 2015 season.

 They also run an A-team (formerly known as "Haringey Hornets") that play in the South Premier Division of the Rugby League Conference.
</item 3>

<item 4>
Wood Green tube station
Wood Green is a London Underground station on the Piccadilly line.

 The station is between Turnpike Lane and Bounds Green stations and is in Travelcard Zone 3.

 It is located at junction of High Road, Wood Green and Lordship Lane.

 It serves Wood Green Shopping City and the nearby Haringey Council administrative complex as well as a densely populated residential area.

 It is also the closest tube station to the Alexandra Palace Venue.
</item 4>

<item 5>
Wood Green Town F.C.
Wood Green Town Football Club was a football club based in Tottenham in the London Borough of Haringey, England.

 Formed in 1911 as a breakaway from Tufnell Park, they merged with Edmonton (a descendent of Tufnell Park) in 1973 to form Edmonton & Haringey.

 During their history the club were known as Tufnell Spartans, Wood Green, Wood Green Town and Haringey Borough.
</item 5>

<item 6>
Haringey Heartlands
Haringey Heartlands is a currently ongoing urban regeneration project in the London Borough of Haringey, in north London, England.

 The site is located west of Shopping City in Wood Green.

 The aim of the development is "to create a vibrant and attractive new urban quarter which acts as the civic and cultural Heart of Haringey, integrating with
and benefiting wider communities."
</item 6>

<item 7>
Alexandra Palace railway station
Alexandra Palace railway station (originally named Wood Green and later Wood Green (Alexandra Park)) is in the London Borough of Haringey in north London, and
is in Zone 3.

 The station, and all trains serving it, are operated by Great Northern.

 Trains mostly go to and from Moorgate on weekdays and weekends.
</item 7>

<item 8>
Haringey London Borough Council
Haringey London Borough Council is the local authority for the London Borough of Haringey in Greater London, England.

 It is a London borough council, one of 32 in the United Kingdom capital of London.

 Haringey is divided into 19 wards, each electing three councillors.

 Haringey London Borough Council comprises 48 Labour Party councillors, and 9 Liberal Democrats.

 The council was created by the London Government Act 1963 and replaced three local authorities: Hornsey Borough Council, Tottenham Borough Council and Wood
Green Borough Council.

 The next election to the authority will be in 2018.
</item 8>

<item 9>
White Hart Lane Community Sports Centre
White Hart Lane Community Sports Centre, also known as the New River Stadium is a rugby league and athletics stadium in Wood Green, Haringey, north London,
England that is home to London Skolars rugby league club, Enfield and Haringey Athletic Club, Wood Green Weightlifting Club, Hashtag United F.C. (YouTube based
football club), Haringey Rhinos rugby union club, Haringey Cycling Club, and the Next Level Football League.
</item 9>

<item 10>
Wood Green War Memorial
The Wood Green War Memorial is located on the High Road, Wood Green, in the London Borough of Haringey.

 It was erected in 1920 and paid for by public subscription of the people of Wood Green to remember the men of the area who died during the First World War.

 It was subsequently adapted to include the dead of the Second World War.

 It is grade II listed with Historic England.
</item 10>
=== END ITEM id=1e6a89e24fd1b4e5 ===

=== ITEM id=684f475f15a77b4b ===
Gold sentences and facts:
Sentence 0: Mulled wine is a beverage usually made with red wine along with various mulling spices and sometimes raisins.
  Fact 0.0: Mulled wine is a beverage.
  Fact 0.1: Mulled wine is usually made with red wine.
  Fact 0.2: Mulled wine is usually made with various mulling spices.
  Fact 0.3: Mulled wine is sometimes made with raisins.
Sentence 1:  It is served hot or warm and is alcoholic, although there are non-alcoholic versions of mulled wine.
  Fact 1.0: Mulled wine is served hot or warm.
  Fact 1.1: Mulled wine is alcoholic.
  Fact 1.2: There are non-alcoholic versions of mulled wine.
Sentence 2: Blue Lagoon is a popular summer cocktail featuring blue Curaçao.
  Fact 2.0: Blue Lagoon is a popular summer cocktail.
  Fact 2.1: Blue Lagoon features blue Curaçao.

Retrieved items:
<item 1>
Mulling spices
Mulling spices is a spice mixture used in drink recipes.

 The spices are usually used to spice hot apple cider, mulled wine, wassail, and other drinks (such as juices) during the autumn or winter.

 A "mulled" drink is a drink which has been prepared with these spices (usually through heating the drink in a pot with mulling spices and then straining).

 Mulling spices may also be added to the brewing process to make spiced beer.
</item 1>

<item 2>
Oz and Hugh Drink to Christmas
Oz and Hugh Drink to Christmas is a BBC television programme in which wine personality Oz Clarke and comedian Hugh Dennis travel through Britain to sample a
wide array of seasonal Christmas beverages, including whisky, winter ales, mulled wine, wassail, sloe gin, Buck's Fizz, Port wine and Sherry.

 Upon its 20 December 2009 broadcast on BBC Two, it had a viewership of approximately 2.4 million with an audience share of 9%.
</item 2>

<item 3>
Volcano bowl
Volcano bowls are ceramic drinkware originally associated with mid-20th century American tiki bars and tropical-themed restaurants.

 Drinks served in volcano bowls are typically rum-based, mixed with tropical fruit juices and other liquors such as brandy, vodka, and triple sec, and garnished
with fruit.

 The Flaming Volcano cocktail is especially associated with this drinkware.
</item 3>

<item 4>
Feuerzangenbowle
Feuerzangenbowle (] ) is a traditional German alcoholic drink for which a rum-soaked sugarloaf is set on fire and drips into mulled wine.

 It is often part of a Christmas or New Year's Eve tradition.

 The name translates literally to "fire-tongs punch".
</item 4>

<item 5>
Blue Lagoon: The Awakening
Blue Lagoon: The Awakening is a 2012 American made-for-television film that premiered on Lifetime on June 16, 2012.

 Indiana Evans and Brenton Thwaites star in the film, which is based on the novel "The Blue Lagoon" and its previous film adaptations.

 Christopher Atkins, the male lead of the 1980 film "The Blue Lagoon", also appears in the film.
</item 5>

<item 6>
The Blue Lagoon (novel)
The Blue Lagoon is a romance novel written by Henry De Vere Stacpoole and was first published by T. Fisher Unwin in 1908.

 It is the first novel of the "Blue Lagoon" trilogy, which also includes "The Garden of God" (1923) and "The Gates of Morning" (1925).

 The novel has inspired several film adaptations, most notably "The Blue Lagoon" starring Brooke Shields as Emmeline and Christopher Atkins as Richard ("Dicky"
in the book).
</item 6>

<item 7>
Blue Lagoon (cocktail)
Blue Lagoon is a popular summer cocktail featuring blue Curaçao.
</item 7>

<item 8>
Flaming volcano
The flaming volcano is a large tropical group cocktail made of rum, brandy, pineapple juice, orange juice, almond syrup, and sometimes other ingredients.

 It is usually a multi-user drink, served to a group in a special, ritual vessel known as a volcano bowl, which is a decorative ceramic bowl (typically of about
32 USoz capacity) designed with a rising central hub feature usually resembling a crude volcano cone.

 The cone includes a second, much smaller central pit or "crater" which is partially filled with rum or other flammable high-alcohol liquor.

 The "crater" liquor is then ignited, creating a mild volcanic ambience with its central blue flame.
</item 8>

<item 9>
Mulled wine
Mulled wine is a beverage usually made with red wine along with various mulling spices and sometimes raisins.

 It is served hot or warm and is alcoholic, although there are non-alcoholic versions of mulled wine.

 It is a traditional drink during winter, especially around Christmas.
</item 9>

<item 10>
Flaming beverage
Flaming beverages include cocktails and other mixed drinks that contain flammable, high-proof alcohol, which is ignited prior to consumption.

 The alcohol may be an integral part of the drink, or it may be floated as a thin layer across the top of the drink.

 The flames are mostly for dramatic flair.

 However, in combination with certain ingredients, the flavor of the drink is altered.

 Some flavors are enhanced, and it may impart a toasted flavor to some drinks.
</item 10>
=== END ITEM id=684f475f15a77b4b ===

=== ITEM id=47acb6298f4fa284 ===
Gold sentences and facts:
Sentence 0: Toshimitsu Deyama (出山利三 , Deyama Toshimitsu , born October 10, 1965 in Tateyama, Chiba) , known exclusively by his stage name Toshi, is a Japanese singer and songwriter.
  Fact 0.0: Toshimitsu Deyama was born October 10, 1965 in Tateyama, Chiba.
  Fact 0.1: Toshimitsu Deyama's Japanese name is 出山利三.
  Fact 0.2: Toshimitsu Deyama's name is also written as Deyama Toshimitsu.
  Fact 0.3: Toshimitsu Deyama is known exclusively by his stage name Toshi.
  Fact 0.4: Toshimitsu Deyama is a Japanese singer and songwriter.
Sentence 1:  He is best known as the lead vocalist and a co-founder of the heavy metal band X Japan, which disbanded in 1997.
  Fact 1.0: Toshimitsu Deyama is best known as the lead vocalist of the heavy metal band X Japan.
  Fact 1.1: Toshimitsu Deyama is a co-founder of X Japan.
  Fact 1.2: X Japan disbanded in 1997.
Sentence 2: Warrel Dane (born March 7, 1969) is the lead singer of the American power metal band Sanctuary.
  Fact 2.0: Warrel Dane was born March 7, 1969.
  Fact 2.1: Warrel Dane is the lead singer of the American power metal band Sanctuary.

Retrieved items:
<item 1>
Sanctuary (band)
Sanctuary is an American heavy metal band founded in Seattle in 1985.

 They split up in 1992, but reformed 18 years later.

 The band consists of Warrel Dane (vocals), Lenny Rutledge (guitar), Nick Cordle (guitar), George Hernandez (bass), and Dave Budbill (drums).

 To date, they have released four studio albums and one live EP.
</item 1>

<item 2>
Nevermore
Nevermore is an American progressive metal band from Seattle, Washington, United States.

 Formed in 1991, they are known to incorporate elements from various styles such as power metal, modern hard rock, classic heavy metal and technical thrash
metal.

 The band has been on hiatus since 2011, due to personal issues between the band members.

 In April 2015, lead singer Warrel Dane confirmed that Nevermore has not disbanded, and there is a possibility for them to continue in the next two years with
another album.
</item 2>

<item 3>
Praises to the War Machine
Praises to the War Machine is the debut and only solo release/album by heavy metal vocalist Warrel Dane from Nevermore, released on April 25, 2008.

 The album was produced by Peter Wichers, who was tapped by Dane to work on the album while Wichers was still a member of Soilwork.

 After leaving that band in 2007, Wichers co-wrote the album with Dane and played on eight of its tracks.

 Soilwork drummer Dirk Verbeuren and former Himsa guitarist Matt Wicklund also played on the album.

 Nevermore guitarist Jeff Loomis and the band's touring guitarist at the time Chris Broderick make guest appearances, along with James Murphy, who had performed
on the Nevermore album "This Godless Endeavor".

 "Praises to the War Machine" includes two cover songs, The Sisters of Mercy's "Lucretia My Reflection" and "Patterns" by Paul Simon.

 Dane had previously covered a Simon song on Nevermore's album "Dead Heart in a Dead World".
</item 3>

<item 4>
The Halo Method
The Halo Method Is an American rock supergroup, formed in 2012 in Hollywood, California.

 It consisted of guitarist Ben Moody (We Are the Fallen, ex-Evanescence), drummer Miles McPherson, vocalist Lukas Rossi (ex-Rock Star Supernova, ex-Daylight
Division) and bassist Josh Newell (ex-In This Moment).
</item 4>

<item 5>
Peter Wichers
Peter Wichers is best known as one of the founding members of Swedish melodic death metal band, Soilwork.

 From 1995 until late 2005, when he announced his departure from the band, he was known as the lead guitarist and also a songwriter, especially on their
"Stabbing the Drama" album., He rejoined Soilwork in 2008 and played on 2010s The Panic Broadcast and in June 2012 he quit Soilwork for the second time.

 After leaving Soilwork, he filled in for Adam Dutkiewicz when he was injured on Killswitch Engage's European tour.

 More recently, he has collaborated with singers from across the melodic death metal genre including Anders Fridén, former band mate Björn "Speed" Strid and
John Bush (vocalist of Anthrax) to make a CD much in the same style as "Roadrunner United", entitled "Nuclear Blast All-Stars".

 He also co-wrote Warrel Dane's solo record "Praises to the War Machine".
</item 5>

<item 6>
Into the Mirror Live
Into The Mirror Live / Black Reflections is live promotional EP of the band Sanctuary recorded on May 12, 1990 at The Country Club in Reseda, California, during
their "Into the Mirror Black" tour.

 Approximately 1000 CD copies of this were made and less than 500 were ever distributed to radio stations and record stores.

 Because of contractual problems with Epic Records, the full version was never released for public consumption.

 Epic Records destroyed 500+ copies of this CD.

 Warrel Dane has stated on several occasions that he has the master tapes and copyrights of the full show and intended to have it finally released.

 To date that has not happened.

 This CD has been heavily pirated over the years as a 2-on-1 CD with the Satan's Host "Metal From Hell" album on the notorious Reborn Classics record label.
</item 6>

<item 7>
Nevermore (album)
Nevermore is the eponymous debut studio album by American heavy metal band Nevermore, released on February 14, 1995 by Century Media Records.

 It was singer Warrel Dane and bassist Jim Sheppard's first release after departing from their previous band, Sanctuary, in 1994.

 Having recruited drummer Van Williams and former Sanctuary touring guitarist Jeff Loomis, they formed the band Nevermore and began anew.
</item 7>

<item 8>
Warrel Dane
Warrel Dane (born March 7, 1969) is the lead singer of the American power metal band Sanctuary.

 He is also known as the former lead singer of currently inactive progressive metal band Nevermore.

 He is a natural baritone, though he was known for his high-pitched vocals with Serpent's Knight and on the first two Sanctuary albums.

 Later in his career, Dane became more notable for his distinctively deep, dramatic voice.
</item 8>

<item 9>
Toshi (musician)
Toshimitsu Deyama (出山利三 , Deyama Toshimitsu , born October 10, 1965 in Tateyama, Chiba) , known exclusively by his stage name Toshi, is a Japanese singer and
songwriter.

 He is best known as the lead vocalist and a co-founder of the heavy metal band X Japan, which disbanded in 1997.

 Toshi then went on to have an extensive solo career.

 In 2007, X Japan reunited and began an ongoing world tour in 2010.

 His distinctive, impassioned tenor vocals have been compared to those of Steve Perry.
</item 9>

<item 10>
Graph query row:
person: Toshimitsu Deyama
organization: None
relationship: None
role: None
period: None
status: None
</item 10>
=== END ITEM id=47acb6298f4fa284 ===

=== ITEM id=a33bece84ebd8afc ===
Gold sentences and facts:
Sentence 0:  He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".
  Fact 0.0: The person described has written a column for Fortune magazine for more than twenty years.
  Fact 0.1: The person described spent a decade at Esquire before writing for Fortune.
Sentence 1:  Founded in 1933, it flourished during the Great Depression under the guidance of founders Arnold Gingrich, David A. Smart and Henry L. Jackson.
  Fact 1.0: The described publication was founded in 1933.
  Fact 1.1: The described publication flourished during the Great Depression.
  Fact 1.2: The described publication flourished under the guidance of founders Arnold Gingrich, David A. Smart, and Henry L. Jackson.

Retrieved items:
<item 1>
VIVmag
VIVmag is one of the earliest exclusively digital, interactive women's lifestyle magazines.

 Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio
Systems, Inc.

 The first issue appeared in January-February 2007.

 There is no print version.

 The paperless title can be viewed in most browsers or downloaded to your computer and read via the Zinio Reader.

 The magazine is based in Thousand Oaks, California.
</item 1>

<item 2>
Esquire (magazine)
Esquire is an American men's magazine, published by the Hearst Corporation in the United States.

 Founded in 1933, it flourished during the Great Depression under the guidance of founders Arnold Gingrich, David A. Smart and Henry L. Jackson.
</item 2>

<item 3>
Simon &amp; Schuster
Simon & Schuster, Inc. ( ), a subsidiary of CBS Corporation, is an American publishing company founded in New York City in 1924 by Richard Simon and Max
Schuster.

 As of 2016, Simon & Schuster publishes 2,000 titles annually under 35 different imprints.
</item 3>

<item 4>
Stanley Bing
Stanley Bing is the pen name of Gil Schwartz (born May 20, 1951 in New York, NY), a business humorist and novelist.

 He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".

 He is the author of thirteen books including "What Would Machiavelli Do?"

 and "The Curriculum", a satirical textbook for a business school that also offers lessons on the Web.

 Schwartz is the senior executive vice president of corporate communications and Chief Communications Officer for CBS.
</item 4>

<item 5>
André Laguerre
Marc André Laguerre (February 21, 1915 – January 18, 1979) was a journalist and magazine editor, best known as the managing editor of "Sports Illustrated" from
1960 to 1974, during which time he oversaw the growth in the magazine from a niche publication to become the industry leader in weekly sports magazines.

 It was under his leadership that the annual "Sports Illustrated Swimsuit Issue" was first published.

 When he retired in 1974, he had been managing editor of the magazine for 704 issues, then a record among magazines published by Time, Inc., SI's parent
company.
</item 5>

<item 6>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 6>

<item 7>
Steve Malzberg
Stephen D. Malzberg (born April 20, 1959) is an American television and radio host, syndicated columnist, and political commentator.

 He is the host of "The Steve Malzberg Show", a cable news and opinion show on Newsmax TV.

 He has also hosted "The Steve Malzberg Show" on various talk radio outlets.
</item 7>

<item 8>
Computer Magazine
as well as "Computer Magazine" staff writers, and is currently managed and edited by the industry veteran Christopher Swearingin an MCSE and former CIO and
regarded author as well as contributor/reporter for "Computer Magazine" and other publications.
</item 8>

<item 9>
This Is Bing Crosby
This Is Bing Crosby was a fifteen-minute five times a week daytime radio program featuring Bing Crosby acting as a disc jockey.

 Minute Maid quick frozen concentrated orange juice was promoted on the shows.
</item 9>

<item 10>
The American Mercury
The American Mercury was an American magazine published from 1924 to 1981.

 It was founded as the brainchild of H. L. Mencken and drama critic George Jean Nathan.

 The magazine featured writing by some of the most important writers in the United States through the 1920s and 1930s.

 After a change in ownership in the 1940s, the magazine attracted conservative writers.

 A second change in ownership a decade later turned the magazine into a virulently anti-Semitic publication.

 It was published monthly in New York City.

 The magazine went out of business in 1981, having spent the last 25 years of its existence in decline and controversy.
</item 10>
=== END ITEM id=a33bece84ebd8afc ===

=== ITEM id=67f73212adb77c78 ===
Gold sentences and facts:
Sentence 0: Ranger Rick, originally Ranger Rick's Nature Magazine, is a children’s nature magazine that is published by the United States National Wildlife Federation.
  Fact 0.0: Ranger Rick was originally called Ranger Rick's Nature Magazine
  Fact 0.1: Ranger Rick is a children's nature magazine
  Fact 0.2: Ranger Rick is published by the United States National Wildlife Federation
Sentence 1: TENNIS is an American sports magazine owned by the Sinclair Broadcast Group.
  Fact 1.0: TENNIS is an American sports magazine
  Fact 1.1: TENNIS is owned by the Sinclair Broadcast Group

Retrieved items:
<item 1>
The entries cover magazines, journalists, writers, and the meaning of “review.” Children’s titles include bimonthly Jack and Jill and National Wildlife
Federation nature magazines Ranger Rick, Ranger Rick Jr., and Ranger Rick Cub. Women’s magazines include Macedonian Tea Moderna and Danish ALT for Damerne.
Other publications range from Adventist Review and conservative National Review to the Human Life Review, Henry James Review, and defunct American Mercury and
Insurgent49. Profiles include Rosie O’Donnell
</item 1>

<item 2>
El Gráfico
El Gráfico is a classic monthly Argentine sports magazine.

 The magazine was originally published as a weekly newspaper in May 1919, and then turned to a sports magazine exclusively.

 It has been scheduled monthly from 2002.

 "El Gráfico" is one of the most regarded sports magazines not only in Argentina but in Latin America.

 The magazine was nicknamed "La Biblia del deporte" ("The Bible of sports") due to its chronicles, notable journalists and collaborators and its photography.
</item 2>

<item 3>
Memphis Sport
Memphis Sport is a sports and fitness magazine featuring articles on local and regional teams, players and events.

 In addition, "Memphis Sport" focuses on health and fitness for an active lifestyle.

 The magazine debuted in Memphis, Tennessee in July 2006.

 Local franchises covered include the University of Memphis Tigers, Memphis Redbirds, Mississippi Riverkings and Tennessee Titans.
</item 3>

<item 4>
Rap-Up
Rap-Up is a magazine launched in 2001 by founder Devin Lazerine.

 The publication was originally a website devoted to hip hop, until Lazerine decided to pitch the possibility of a magazine to several publishers.

 The magazine is focused on the hip hop and R&B aspect of the music industry, and predominantly features interviews with artists, actors and other entertainers.

 Issued quarterly, the magazine's target audience was the 14 to 28 demographic, although the demographic has become older over time.
</item 4>

<item 5>
Lorin Thompson
Lorin Thompson, Lorin Hartwell Jr Thompson, (1911–1997), was a muralist, artist, and creator and illustrator of the character "Ranger Rick" for the National
Wildlife Federation's children's magazine, Ranger Rick.
</item 5>

<item 6>
Ranger Rick
Ranger Rick, originally Ranger Rick's Nature Magazine, is a children’s nature magazine that is published by the United States National Wildlife Federation.

 The magazine offers feature articles and activities for children, ages 7 and up, in order to spark their interest in the outdoors and become more actively
involved in the environment.

 The magazine's primary intention is to instill a passion for nature and promote activity outdoors.

 NWF also publishes two companion magazines, "Ranger Rick Jr.", which is aimed at ages 4–7, and "Ranger Rick Cub", which is aimed at kids 0–4 years old.)
</item 6>

<item 7>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
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
Tennis (magazine)
TENNIS is an American sports magazine owned by the Sinclair Broadcast Group.

 There are eight issues per year covering the world of tennis.

 The "tennis.com" website is an offshoot of the magazine and part of the same media company.
</item 9>

<item 10>
Football Weekly is The Guardian’s football podcast: Mondays, with Thursday spin-off Football Weekly Extra; it aired daily during major tournaments from Euro
2008 to Euro 2016. Football World, founded in 1921, became Athletic World and then Outing in 1924. World Soccer, established in 1960, covers international
football and belongs to the European Sports Magazines group. TENNIS publishes eight issues yearly. Argentina’s El Gráfico, founded in 1919, is
</item 10>
=== END ITEM id=67f73212adb77c78 ===

=== ITEM id=d35bd456644fbfcd ===
Gold sentences and facts:
Sentence 0:  Programming is simulcast on WANH (91.5 FM) in Meredith, serving the Lakes Region.
  Fact 0.0: Programming is simulcast on WANH (91.5 FM).
  Fact 0.1: WANH (91.5 FM) is located in Meredith.
  Fact 0.2: WANH serves the Lakes Region.
Sentence 1:  The area comprises all of Belknap County, the southern portion of Carroll County, the eastern portion of Grafton County, and the northern portions of Strafford County and Merrimack County.
  Fact 1.0: The area comprises all of Belknap County.
  Fact 1.1: The area comprises the southern portion of Carroll County.
  Fact 1.2: The area comprises the eastern portion of Grafton County.
  Fact 1.3: The area comprises the northern portion of Strafford County.
  Fact 1.4: The area comprises the northern portion of Merrimack County.

Retrieved items:
<item 1>
KRBT
KRBT (1340 AM) is a radio station broadcasting a sports format.

 Licensed to Eveleth, Minnesota, United States, the station serves the Iron Range area.

 The station is currently owned by Range Broadcasting, Inc., and features programming from the Red Zone Radio Network.

 Originally a stand alone music station with call letters WEVE, then a simulcast music station with WEVE-FM, the station changed it's call letters to KRBT for
"Range's Best Talk" in June of 1998 and discontinued simulcast with WEVE-FM to become a talk station.

 The format change to sports/talk was made following the death of previous owner, Lew Latto, Iron Range Broadcasting, Inc.

 President on August, 24 2011 and subsequent sale to Red Rock Radio in March 2012, and has been retained by the new owner as of July 2017, Range Broadcasting,
Inc.
</item 1>

<item 2>
WVNH
WVNH (91.1 FM) is a radio station broadcasting a Christian radio format.

 Licensed to Concord, New Hampshire, United States, the station serves the Concord area.

 The station is currently owned by New Hampshire Gospel Radio, Inc.

 Programming is simulcast on WANH (91.5 FM) in Meredith, serving the Lakes Region.
</item 2>

<item 3>
WLMR
WLMR (1450 AM) is a radio station broadcasting a religious format.

 Licensed to Chattanooga, Tennessee, USA, the station serves the Chattanooga area.

 The station is currently owned by Wilkins Communications Network, Inc. and features programming from USA Radio Network.

 In the early 1980s, the station was automated and played country music from studios in a strip mall on Brainerd Rd.

 It was also Chattanooga's first talk radio station when it held the WZRA call letters.

 WZRA was the first home of Jeff Styles, and also featured well known personalities such as Kelly McCoy and Robert T. Nash.
</item 3>

<item 4>
KVKI-FM
KVKI-FM (96.5 FM) is a radio station broadcasting an adult contemporary music format.

 Licensed to Shreveport, Louisiana, United States, the station serves the Shreveport area.

 The station is currently owned by Townsquare Media.

 Its studios are shared with its other five sister stations in West Shreveport (one mile west of Shreveport Regional Airport), and the transmitter is in
Blanchard, Louisiana.

 The call letters were originally KBCL, and later became religious station KEPT, before the call letters were changed to KVKI to be similar to the call letters
of the popular KVIL radio station in Dallas, Texas.
</item 4>

<item 5>
Lakes Region (New Hampshire)
The Lakes Region of New Hampshire is the mid-state region surrounding Lake Winnipesaukee, Winnisquam Lake, Squam Lake, and Newfound Lake.

 The area comprises all of Belknap County, the southern portion of Carroll County, the eastern portion of Grafton County, and the northern portions of Strafford
County and Merrimack County.

 The largest municipality is the city of Laconia.
</item 5>

<item 6>
KRBT (1340 AM, Eveleth, MN), owned by Range Broadcasting, airs Red Zone sports programming; formerly WEVE, it became talk radio in 1998 and sports/talk after
ownership changes in 2011–12. KWXL-LP (98.7 FM, Tucson), Tucson Unified School District’s high-school variety/news station, serves Pueblo students. KQIP-LP
(107.1 FM, Chico), owned by Calvary Chapel, broadcasts Bible
</item 6>

<item 7>
KMBC-TV
KMBC-TV, virtual channel 9 (UHF digital channel 29), is the ABC-affiliated television station licensed to Kansas City, Missouri, United States and also serving
Kansas City, Kansas.

 The station is owned by the Hearst Television subsidiary of the Hearst Corporation, as part of a duopoly with CW affiliate KCWE (channel 29).

 The two stations share studio facilities located at the Winchester Business Center on Winchester Avenue and East 63rd Street (near Swope Park, off of I-435) in
southeastern Kansas City, Missouri; KMBC-TV maintains transmitter facilities located near the Blue River in eastern Kansas City.

 On cable, KMBC-TV is available on Charter Spectrum, Comcast Xfinity and Consolidated Communications channel 12, and Google Fiber and AT&T U-verse channel 9.

 There is a high definition feed provided on Spectrum digital channel 1200, Xfinity channel 812, Consolidated channel 610 and U-verse channel 1009.
</item 7>

<item 8>
KKBN
KKBN (93.5 FM) The Cabin a radio station since 1986 broadcasting a Country music format since March 2000.

 Licensed to Twain Harte, California, United States, the station serves the greater Mother Lode area comprising Tuolumne and Calaveras Counties with a combined
population of approximately 110,000.

 The station is currently owned by Clarke Broadcasting Corporation.
</item 8>

<item 9>
- **KHSL-TV:** CBS station serving Chico and Redding, California; Heartland Media-owned, paired with NBC affiliate KNVN, and based in Chico.
- **WXMK:** Hot Adult Contemporary station, 105.9 FM, serving Brunswick, Georgia; owned by Golden Isles Broadcasting.
- **WLMR:** Religious-format AM station serving Chattanooga, Tennessee; owned by Wilkins Communications, with USA Radio Network programming. Formerly country
and an early talk-radio station.
-
</item 9>

<item 10>
Spectrum News Rochester and Buffalo are Charter-owned 24-hour regional news channels covering Rochester/Finger Lakes and Buffalo/Western New York; Buffalo also
supplies Jamestown’s mixed-content Cable 8 News feed. Milwaukee’s WDJT-TV is a Weigel-owned CBS station in a four-station group. Chicago’s Fox-owned WFLD is
paired with WPWR-TV. Jacksonville’s Tegna-owned NBC affiliate WTLV and ABC affiliate WJXX share studios and form a duopoly
</item 10>
=== END ITEM id=d35bd456644fbfcd ===

=== ITEM id=1731448b3d6ed70f ===
Gold sentences and facts:
Sentence 0:  Reed, along with Mike Pringle and Johnny Bright, is one of the players most often mentioned as being the greatest running back in CFL history.
  Fact 0.0: Reed is one of the players most often mentioned as being the greatest running back in CFL history.
  Fact 0.1: Mike Pringle is one of the players most often mentioned as being the greatest running back in CFL history.
  Fact 0.2: Johnny Bright is one of the players most often mentioned as being the greatest running back in CFL history.
Sentence 1: Johnny D. Bright (June 11, 1930 – December 14, 1983) was a professional Canadian football player in the Canadian Football League.
  Fact 1.0: Johnny D. Bright was born June 11, 1930.
  Fact 1.1: Johnny D. Bright died December 14, 1983.
  Fact 1.2: Johnny D. Bright was a professional Canadian football player.
  Fact 1.3: Johnny D. Bright played in the Canadian Football League.
Sentence 2:  He played college football at Drake University.
  Fact 2.0: Johnny D. Bright played college football at Drake University.

Retrieved items:
<item 1>
Dolph Pulliam
Adolphus "Dolph" Pulliam (born March 14, 1946 in West Point, Mississippi) is a former American basketball player.

 He played collegiately at Drake University.

 He was named to Drake University's All-Century basketball team.

 He helped lead the Drake men's basketball team to the 1969 Final Four and a third-place finish.

 On 7 February 2009, Drake University retired Pulliam's #5 jersey that he wore for the Bulldogs.
</item 1>

<item 2>
George Reed (Canadian football)
George Robert Reed, CM, SOM (born October 2, 1939), is a former American college football and Canadian Football League player.

 Reed, along with Mike Pringle and Johnny Bright, is one of the players most often mentioned as being the greatest running back in CFL history.

 In November 2006, Reed was voted one of the CFL's Top 50 players (#2) of the league's modern era by Canadian sports network TSN.
</item 2>

<item 3>
Damon Allen
Damon Allen (born July 29, 1963) is a former professional quarterback who played in the Canadian Football League.

 He is currently second in all-time professional football passing yards and second in all-time CFL passing yards after he was surpassed for first place by the
Montréal Alouettes' Anthony Calvillo on October 10, 2011.

 He is pro-football's combined passing & rushing leader with a total of 84,301 yards.

 Allen retired as professional football's all-time leading passer with 72,381 passing yards after he surpassed Warren Moon's total of 70,553 yards (in both the
CFL and NFL combined) on September 4, 2006 in the annual Labour Day Classic.

 He also retired in third place in all-time CFL rushing yards with 11,920 yards, behind Mike Pringle and George Reed.

 The 2007 season marked Allen's twenty-third season in the CFL and he officially announced his retirement on May 28, 2008 at age 44.

 Allen is the younger brother of Pro Football Hall of Famer Marcus Allen.
</item 3>

<item 4>
Terry Baker
Terry Wayne Baker (born May 5, 1941) is a former American football and basketball player.

 He played college football and college basketball at the Oregon State University.

 He played as a quarterback for the Oregon State Beavers football team from 1960 to 1962, winning the Heisman Trophy as senior.

 In the spring of his senior year, he played in the Final Four of the 1963 NCAA Men's Division I Basketball Tournament with the Oregon State Beavers men's
basketball team.

 To date, he is the only athlete to win a Heisman Trophy and play in the Final Four.

 Baker was the first overall pick in the 1963 NFL draft and played with the Los Angeles Rams of the National Football League (NFL) from 1963 to 1965.

 He then played for one season in the Canadian Football League (CFL) with the Edmonton Eskimos, in 1966.

 Baker was inducted into the College Football Hall of Fame in 1982.
</item 4>

<item 5>
Richard Harris (American football)
Richard Drew Harris (January 21, 1948 – July 26, 2011) was an American football defensive end who played seven seasons in the National Football League.

 He was and All-American in 1970 for Grambling and was drafted in the first round (5th overall pick) of the 1971 NFL Draft by the Philadelphia Eagles.

 He was named All-Rookie in 1971, playing defensive end.

 Harris spent seven seasons as a defensive end in the NFL, the first three with the Philadelphia Eagles, the next two with the Chicago Bears and the final two
with the Seattle Seahawks.
</item 5>

<item 6>
Mike Hogan (sportscaster)
Mike Hogan (born 1963 in Kingston, Ontario) is a Canadian sportscaster who is a sports talk radio host on TSN Radio 1050 in Toronto.

 He also serves as the play-by-play voice for the Toronto Argonauts of the Canadian Football League for that station's gameday broadcasts.

 Hogan was formerly a long time sports talk radio host on Toronto station CJCL (AM), branded "The Fan 590", until he was relieved of those duties on June 24,
2010.

 On 2017 Oscar Vanity fair party 1st live broadcast Mike Hogan was the podcast hosts.
</item 6>

<item 7>
Marc Mueller
Marc Mueller (born April 15, 1989) is the current running backs coach of the Calgary Stampeders of the Canadian Football League (CFL).

 In college, he played quarterback for the Regina Rams of the CIS.

 He signed as an undrafted free agent with the Edmonton Eskimos of the CFL on May 9, 2011 following the 2011 CFL Draft.

 He was released on June 20, 2011 in order to play his final year for the Rams.

 Mueller is the grandson of former CFL quarterback and coach Ron Lancaster.
</item 7>

<item 8>
Johnny Bright
Johnny D. Bright (June 11, 1930 – December 14, 1983) was a professional Canadian football player in the Canadian Football League.

 He played college football at Drake University.

 He is a member of the Canadian Football Hall of Fame, the National Football Foundation's College Football Hall of Fame, the Missouri Valley Conference Hall of
Fame, the Edmonton Eskimos Wall of Honour, the Alberta Sports Hall of Fame, and the "Des Moines Register's" Iowa Sports Hall of Fame.
</item 8>

<item 9>
Paul Hornung
Paul Vernon Hornung (born December 23, 1935), nicknamed "The Golden Boy", is a former professional American football player, a Hall of Fame running back for the
Green Bay Packers of the National Football League (NFL) from 1957 to 1966, winning four NFL titles and the first Super Bowl.

 He is the first in pro football history to win the Heisman Trophy, be selected as the first overall selection in the NFL Draft, win the NFL most valuable
player award, and be inducted into both the professional and college football halls of fame.
</item 9>

<item 10>
2,000-yard club
The 2,000-yard club is a group of seven National Football League (NFL) running backs that have rushed for 2,000 or more yards in a season.

 These seven rushing seasons rank as the highest single-season rushing totals in NFL history, and reaching the 2,000-yard mark is considered a significant
achievement for running backs.

 No running back has yet achieved this feat twice.

 The first 2,000-yard season was recorded in 1973 by Buffalo Bills running back O.J. Simpson.

 He is the only player to have surpassed 2,000 yards in a 14-game season, as all others occurred in 16-game seasons; he finished the season with 2,003 rushing
yards, averaging six yards per carry and an NFL-record 143.1 rushing yards per game.

 Los Angeles Rams running back Eric Dickerson, who had broken the single-season rookie rushing record in 1983, recorded the second 2,000-yard season in 1984.

 Dickerson rushed for 2,105 yards, the current NFL rushing record, and averaged 131.6 rushing yards per game.
</item 10>
=== END ITEM id=1731448b3d6ed70f ===
