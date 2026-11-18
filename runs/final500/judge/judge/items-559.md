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

=== ITEM id=b007eb128af389aa ===
Gold sentences and facts:
Sentence 0: The Random House Tower, also known as the Park Imperial Apartments, is a 52-story mixed-use tower in New York City, United States, that is used as the headquarters of book publisher Random House and a luxury apartment complex.
  Fact 0.0: The Random House Tower is also known as the Park Imperial Apartments.
  Fact 0.1: The Random House Tower is a 52-story mixed-use tower.
  Fact 0.2: The Random House Tower is in New York City, United States.
  Fact 0.3: The Random House Tower is used as the headquarters of book publisher Random House.
  Fact 0.4: The Random House Tower is used as a luxury apartment complex.
Sentence 1:  It currently carries the Vornado Realty Trust corporate headquarters.
  Fact 1.0: The Random House Tower currently carries the Vornado Realty Trust corporate headquarters.

Retrieved items:
<item 1>
Random House Tower
The Random House Tower, also known as the Park Imperial Apartments, is a 52-story mixed-use tower in New York City, United States, that is used as the
headquarters of book publisher Random House and a luxury apartment complex.

 The book publisher entrance is on Broadway and goes up to 27 floors, while the apartment complex entrance is on West 56th Street.
</item 1>

<item 2>
655 Park Avenue
655 Park Avenue is a Georgian-style co-op residential building on Manhattan's Upper East Side, located on Park Avenue between 67th Street and 68th Street,
adjacent to the Park Avenue Armory.

 It was developed in 1924 by Dwight P. Robinson & Company.

 The building at 655 Park Avenue was designed by architects James Edwin Ruthven Carpenter, Jr., often referred to by the initials "J.E.R. Carpenter", and Mott
B. Schmidt.

 Carpenter is considered the leading architect for luxury residential high-rise buildings in New York City in the early 1900s, while Schmidt is known for his
buildings in the American Georgian Classical style, including Sutton Place and houses for New York City's society figures and business elite.
</item 2>

<item 3>
888 7th Avenue
888 7th Avenue is a 628 ft (191m) tall modern-style office skyscraper in Midtown Manhattan which was completed in 1969 and has 46 floors.

 Emery Roth & Sons designed the building, which is tied with Central Park Place for the 65th tallest building in New York City.

 It currently carries the Vornado Realty Trust corporate headquarters.

 Previously known as the Arlen Building, its namesake being the company responsible for its construction, Arlen Realty & Development Corporation.

 The Red Eye Grill is located in the building at street level.
</item 3>

<item 4>
245 Park Avenue
245 Park Avenue (formerly American Tobacco Company Building, American Brands Building and Bear Stearns Building) is a 648-ft (198 m) tall skyscraper in New York
City, New York.

 It was completed in 1967 and has 48 floors (1.7 million square feet).

 Shreve, Lamb and Harmon designed the building.

 The Building Owners and Managers Association awarded the 2000/2001 Pinnacle Award to 245 Park Avenue.

 The site used to be occupied by the second Grand Central Palace exhibition hall, which was demolished in 1964 to make way for 245 Park Avenue.
</item 4>

<item 5>
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
</item 5>

<item 6>
Transportation Building
The Transportation Building is a 44-story office building located at 225 Broadway on the corner of Barclay Street in the Civic Center neighborhood of Lower
Manhattan in New York City.

 It also carries the address 2-4 Barclay Street.

 It was built in 1927 and was designed by the architecture firm of York & Sawyer, in the Renaissance Revival style, using setbacks common to skyscrapers built
after the adoption of the 1916 Zoning Resolution.

 It sits across Barclay Street from the Woolworth Building.
</item 6>

<item 7>
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
</item 7>

<item 8>
383 Madison Avenue
383 Madison Avenue is an office building owned and occupied by JP Morgan Chase in New York City on a full block bound by Madison Avenue and Vanderbilt Avenue
between East 46th and 47th Streets.

 Formerly known as the Bear Stearns Building, it housed the world headquarters of the now-defunct Bear Stearns from the building's completion until Bear's
collapse and sale to JPMorgan Chase in 2008.

 The building now houses the New York offices for J.P. Morgan's investment banking division, which formerly occupied 277 Park Avenue.

 Both 383 Madison and 277 Park are adjacent to JPMorgan Chase's world headquarters at 270 Park Avenue.
</item 8>

<item 9>
The entries cover notable U.S. buildings, districts, businesses, and people. New York features prominently: office towers such as 888 Seventh Avenue, 245 Park
Avenue, 383 and 270 Madison/Park Avenue, and Goldman Sachs’ 200 West Street; landmarks including the Woolworth Building and 90 West Street; and residential or
mixed-use towers such as MiMA and Random House Tower. Other topics include historic theaters, retail and media companies, transportation firms, and real-estate
developer
</item 9>

<item 10>
Rubin Schron
Other buildings he owns include the Monterey, a 521-unit rental multifamily building on Manhattan's Upper East Side; over the decades, Schron has also amassed a
portfolio of Mitchell-Lama apartment buildings whose values have been skyrocketing to record values, after reverting to market rates when government subsidies
expired.

 In 2007, he sold nearly 4,000 units of former Mitchell-Lama properties in five complexes in Harlem and on Roosevelt Island for $940 million.

 Schron, who practices Orthodox Judaism, has eight children and 50 grandchildren.

 He and his family have lived in the same single-family home in Brooklyn for many decades.
</item 10>
=== END ITEM id=b007eb128af389aa ===

=== ITEM id=4086f4f2cdc1c828 ===
Gold sentences and facts:
Sentence 0:  Formed by Corgan and guitarist James Iha in Chicago, Illinois, in 1988, the band quickly gained steam with the addition of bassist D'arcy Wretzky and drummer Jimmy Chamberlin.
  Fact 0.0: The band was formed by Corgan and guitarist James Iha.
  Fact 0.1: The band was formed in Chicago, Illinois, in 1988.
  Fact 0.2: The band quickly gained steam with the addition of bassist D'arcy Wretzky.
  Fact 0.3: The band quickly gained steam with the addition of drummer Jimmy Chamberlin.
Sentence 1: Greek Fire is an American rock band from St. Louis, Missouri.
  Fact 1.0: Greek Fire is an American rock band.
  Fact 1.1: Greek Fire is from St. Louis, Missouri.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Pinhead Gunpowder
Pinhead Gunpowder is an American punk rock band that formed in East Bay, California, in 1990.

 The band currently consists of Aaron Cometbus (drums, lyrics), Bill Schneider (bass), Billie Joe Armstrong (guitar, vocals) and Jason White (guitar, vocals).

 The band's name comes from a brand of "high octane" green tea served at the Arcata co-op and discovered by Aaron Cometbus during one of his many dumpster
diving adventures.
</item 2>

<item 3>
Greek Fire (band)
Greek Fire is an American rock band from St. Louis, Missouri.

 The band was formed in 2008 by members of Story of the Year and Maybe Today.

 Since formation, Greek Fire has released a self-titled EP, a single titled "Doesn't Matter Anyway", on August 16, 2011, they released their debut, full-length
album, "Deus Ex Machina", and have recently announced a new addition to the "Lost/Found" EPs titled "Broken" set to be released before "Found".
</item 3>

<item 4>
1979 (song)
"1979" is a song by American alternative rock band the Smashing Pumpkins.

 Released in 1996 as the second single from their third studio album, "Mellon Collie and the Infinite Sadness", "1979" was written by frontman Billy Corgan, and
features loops and samples that were uncharacteristic of previous Smashing Pumpkins songs.

 The song was written as a coming of age story by Corgan.

 In the year 1979, Corgan was 12 and this is what he considered his transition into adolescence.

 The song was popular with critics and fans; Allmusic's Amy Hanson called it a "somewhat surprising hit".

 The song was nominated for the Record of the Year and Best Rock Performance by a Duo or Group with Vocal at the 39th Annual Grammy Awards, and won an MTV Video
Music Award for Best Alternative Video.

 In 2012, it was voted the second-best Smashing Pumpkins song by Rolling Stone magazine readers.
</item 4>

<item 5>
Muzzle (song)
"Muzzle" is a song by American alternative rock band The Smashing Pumpkins from their third album, "Mellon Collie and the Infinite Sadness".

 It was one of the last songs written by Billy Corgan for "Mellon Collie", with the song's lyrics referring to what Corgan thought the public's perception was
of him at the time.

 It was rumored to be the Smashing Pumpkins fifth and final single from this album, as is evidenced by the fact that a promotional single for the song was
issued to radio stations worldwide.

 However, the song "Thirty-Three" was released as the fifth and final single instead.
</item 5>

<item 6>
List of the Smashing Pumpkins band members
The Smashing Pumpkins are an alternative rock band formed in Chicago, Illinois in 1988.

 The band was formed by guitarist/vocalist Billy Corgan and guitarist James Iha after the demise of Corgan's first band, The Marked.

 Since its inception, The Smashing Pumpkins has gone through multiple line-up changes, with Corgan the only consistent member.
</item 6>

<item 7>
Zeitgeist (The Smashing Pumpkins album)
Zeitgeist is the seventh album by American alternative rock band The Smashing Pumpkins, released on July 10, 2007 in the United States and Canada.

 It was the first album The Smashing Pumpkins released after their 2000 disbandment and 2005 reunion.

 The album was produced by Roy Thomas Baker, Billy Corgan, Jimmy Chamberlin, and Terry Date.

 The album would be Chamberlin's last with the band before his departure in 2009.

 The album debuted strongly, but sales soon decreased, and critical reception was mixed.

 It was certified Gold in the United States on February 1, 2008.
</item 7>

<item 8>
Conspiracy of Fire Nuclei
The Conspiracy of Fire Nuclei (Greek: Συνωμοσία των Πυρήνων της Φωτιάς , "Synomosía Pyrínon Tis Fotiás" –SPF), also translated as Conspiracy of Fire Cells or
Conspiracy of Cells of Fire, is a radical anarchist organization based in Greece.

 The SPF first surfaced on January 21, 2008, with a wave of 11 firebombings against luxury car dealerships and banks in Athens and Thessaloniki.

 Monthly waves of arson have been followed by proclamations expressing solidarity with arrested anarchists in Greece and elsewhere.

 In September 2009, following an escalation to the use of crude time bombs, four suspected members of the group were arrested.

 In November 2010 two more suspects were arrested while attempting to mail parcel bombs to embassies and EU leaders and organizations.

 The organization was designated as a terrorist group on October 13, 2011, by the U.S. Department of State.
</item 8>

<item 9>
Billy Corgan
William Patrick "Billy" Corgan Jr. (born March 17, 1967) is an American musician, songwriter, producer, poet, and entrepreneur.

 He is best known as the lead singer, primary songwriter, guitarist, and sole permanent member of The Smashing Pumpkins.

 Formed by Corgan and guitarist James Iha in Chicago, Illinois, in 1988, the band quickly gained steam with the addition of bassist D'arcy Wretzky and drummer
Jimmy Chamberlin.

 Strong album sales and large-scale tours propelled the band's increasing fame in the 1990s until their break-up in 2000.

 Corgan started a new band called Zwan, and after their quick demise, he released a solo album ("TheFutureEmbrace") and a collection of poetry ("Blinking with
Fists") before setting his sights on reforming Smashing Pumpkins.
</item 9>

<item 10>
Mike Byrne (musician)
Michael William Byrne (born February 6, 1990) is a drummer who was a member of the band The Smashing Pumpkins.

 When Pumpkins frontman Billy Corgan called for rehearsals to replace Jimmy Chamberlin, Byrne auditioned and was chosen out of thousands of applicants.

 He played drums in Sky Saxon tribute band Spirits in the Sky for 6 shows in August 2009, along with Corgan, Kerry Brown, Kevin Dippold, Mark Tulin, Linda
Strawberry, Ysanne Spevack, Mark Weitz, and Dave Navarro, and also performs drums on the ongoing Pumpkins 44-song project, "Teargarden by Kaleidyscope".

 As a part of "Teargarden", Mike contributed drumming duties on "Oceania", the Pumpkins' eighth full-length album.

 On April 17, 2010, Byrne played his first show under the Smashing Pumpkins moniker in celebration of Record Store Day.
</item 10>
=== END ITEM id=4086f4f2cdc1c828 ===

=== ITEM id=bd6f510c52ede252 ===
Gold sentences and facts:
Sentence 0: "Collateral Damage" is the eleventh episode of the third season of the crime-thriller television series "Millennium".
  Fact 0.0: 'Collateral Damage' is the eleventh episode of the third season of the crime-thriller television series 'Millennium'.
  Fact 0.1: 'Millennium' is a crime-thriller television series.
  Fact 0.2: 'Collateral Damage' is in the third season of 'Millennium'.
Sentence 1:  "Collateral Damage" featured guest appearances by O'Quinn, Jacinda Barrett, James Marsters, and radio host Art Bell as himself.
  Fact 1.0: 'Collateral Damage' featured a guest appearance by O'Quinn.
  Fact 1.1: 'Collateral Damage' featured a guest appearance by Jacinda Barrett.
  Fact 1.2: 'Collateral Damage' featured a guest appearance by James Marsters.
  Fact 1.3: 'Collateral Damage' featured a guest appearance by radio host Art Bell.
  Fact 1.4: Art Bell appeared as himself in 'Collateral Damage'.
Sentence 2:  Bell's radio show "Coast to Coast AM" was among Perry's inspirations for the script.
  Fact 2.0: Art Bell's radio show 'Coast to Coast AM' was among Perry's inspirations for the script.
Sentence 3: Coast to Coast AM is an American late-night radio talk show that deals with a variety of topics.
  Fact 3.0: Coast to Coast AM is an American late-night radio talk show.
  Fact 3.1: Coast to Coast AM deals with a variety of topics.
Sentence 4:  The program now airs seven nights a week 1:00 a.m. – 5:00 a.m. Eastern Time Zone.
  Fact 4.0: Coast to Coast AM now airs seven nights a week.
  Fact 4.1: Coast to Coast AM airs from 1:00 a.m. to 5:00 a.m. Eastern Time Zone.

Retrieved items:
<item 1>
Night After Night with Allan Havey
Night After Night with Allan Havey was a late night talk show which aired on The Comedy Channel and, later, Comedy Central from 1989 to 1992.

 The show featured the stream of consciousness of host Allan Havey, and other departures from the standard late-night talk show format.

 Havey's sidekick was Nick Bakay, who moved on to act as sidekick on Dennis Miller's short-lived late night talk show.
</item 1>

<item 2>
Coast to Coast AM
Coast to Coast AM is an American late-night radio talk show that deals with a variety of topics.

 Most frequently the topics relate to either the paranormal or conspiracy theories.

 The program is distributed by Premiere Networks, both as part of its talk network and separately as a syndicated program.

 The program now airs seven nights a week 1:00 a.m. – 5:00 a.m. Eastern Time Zone.
</item 2>

<item 3>
The Week Show
The Week Show is a television late night talk show sketch comedy program on Bite TV in Canada.

 It originally ran from December 2008 to mid-2010.

 The show is hosted by Matt Chin and Ricky Thompson, who started on an internet late night talk show "My Show with Matt Chin" online in 2001.
</item 3>

<item 4>
Steve Malzberg
Stephen D. Malzberg (born April 20, 1959) is an American television and radio host, syndicated columnist, and political commentator.

 He is the host of "The Steve Malzberg Show", a cable news and opinion show on Newsmax TV.

 He has also hosted "The Steve Malzberg Show" on various talk radio outlets.
</item 4>

<item 5>
Pilot (Studio 60 on the Sunset Strip)
"Pilot" is the first episode of the television series "Studio 60 on the Sunset Strip".

 The episode was first aired in the United States on the NBC network on September 18, 2006.

 Written by series creator Aaron Sorkin, and directed by executive producer Thomas Schlamme, the episode introduces the chaotic behind-the-scenes depiction of a
fictional "Saturday Night Live" type show also called "Studio 60 on the Sunset Strip".
</item 5>

<item 6>
David Letterman
David Michael Letterman (born April 12, 1947) is an American television host, comedian, writer, and producer.

 He hosted a late night television talk show for 33 years, beginning with the February 1, 1982 debut of "Late Night with David Letterman" on NBC, and ending
with the May 20, 2015 broadcast of "Late Show with David Letterman" on CBS.

 In total, Letterman hosted 6,028 episodes of "Late Night" and "Late Show", surpassing friend and mentor Johnny Carson as the longest-serving late night talk
show host in American television history.

 In 1996 Letterman was ranked 45th on "TV Guide"' s 50 Greatest TV Stars of All Time.
</item 6>

<item 7>
Morning Joe
Morning Joe is a weekday NBC News morning news and talk show, airing from 6:00 a.m. to 9:00 a.m. Eastern Time on the network's cable news channel MSNBC.

 It features former GOP representative Joe Scarborough providing both enterprise reporting and discussion on the news of the day in a panel format with co-hosts
Mika Brzezinski and Willie Geist.

 It was created as the replacement for "Imus in the Morning," which was cancelled in April 2007 after simulcasting on MSNBC since 1996.
</item 7>

<item 8>
Collateral Damage (Millennium)
"Collateral Damage" is the eleventh episode of the third season of the crime-thriller television series "Millennium".

 It premiered on the Fox network on January 22, 1999.

 The episode was written by Michael R. Perry, and directed by Thomas J. Wright.

 "Collateral Damage" featured guest appearances by O'Quinn, Jacinda Barrett, James Marsters, and radio host Art Bell as himself.

 Bell's radio show "Coast to Coast AM" was among Perry's inspirations for the script.
</item 8>

<item 9>
Late-night talk show
A late-night talk show is a genre of talk show popular in the United States, where the format originated.

 It is generally structured around humorous monologues about the day's news, guest interviews, comedy sketches and music performances.

 The late-night talk show format was popularized, though not invented, by Johnny Carson with "The Tonight Show Starring Johnny Carson" on NBC.

 Typically the show's host conducts interviews from behind a desk, while the guest is seated on a couch.

 Many late night talk shows feature a house band which generally performs cover songs for the studio audience during commercial breaks and occasionally will
back up a guest artist.
</item 9>

<item 10>
CNN World News
CNN World News, a program that airs on CNN International and CNN International Asia Pacific.

 It is supplemented by CNN World News Asia and CNN World News Europe The show's traditional time run is 24-hours if it is followed by CNN World News Middle East
The show's regular presenters include Errol Barnett.

 Its main role is to update viewers of the latest news in the world.

 It contains a weather update from the CNN World Weather Forecast News.

 CNN World News can usually air up to three times on weekends, and is known to be bringing the latest on a story.
</item 10>
=== END ITEM id=bd6f510c52ede252 ===

=== ITEM id=e64f7610e1be1d90 ===
Gold sentences and facts:
Sentence 0: Marc Rosset (born 7 November 1970) is a former professional tennis player from Switzerland who is best remembered for winning the men's singles gold medal at the 1992 Olympic Games.
  Fact 0.0: Marc Rosset was born on 7 November 1970.
  Fact 0.1: Marc Rosset is a former professional tennis player.
  Fact 0.2: Marc Rosset is from Switzerland.
  Fact 0.3: Marc Rosset is best remembered for winning the men's singles gold medal at the 1992 Olympic Games.
Sentence 1: Maksim "Max" Mirnyi (Belarusian: Максім Мікалаевіч Мірны ] ) (Russian: Максим Николаевич Мирный ] ) (born 6 July 1977, in Minsk) is a Belarusian professional tennis player.
  Fact 1.0: Maksim Mirnyi, also known as Max Mirnyi, was born on 6 July 1977 in Minsk.
  Fact 1.1: Maksim Mirnyi's Belarusian name is Максім Мікалаевіч Мірны.
  Fact 1.2: Maksim Mirnyi's Russian name is Максим Николаевич Мирный.
  Fact 1.3: Maksim Mirnyi is a Belarusian professional tennis player.

Retrieved items:
<item 1>
Marat Safin
Marat Mubinovich Safin (Russian: Мара́т Муби́нович Са́фин ; ] , Tatar: Марат Мөбин улы Сафин "Marat Möbin uğlı Safin" ; born 27 January 1980) is a Russian
politician and retired professional tennis player.

 He achieved the Association of Tennis Professionals (ATP) world No. 1 singles ranking on November 20, 2000.

 Nobody taller than Safin has ever reached his world number-one ranking.

 Safin is the older brother of former world No. 1 WTA player, Dinara Safina.

 They are the first and only brother–sister tandem in tennis history who have both achieved No. 1 rankings.

 On court, Safin was famous for his emotional outbursts and sometimes fiery temper.
</item 1>

<item 2>
2005 BNP Paribas Masters – Doubles
Jonas Björkman and Todd Woodbridge were the defending champions, but did not participate together this year.

 Björkman partnered Max Mirnyi, losing in the first round.

 Woodbridge retired from professional tennis earlier in the year.
</item 2>

<item 3>
Michael Venus (tennis)
Michael Venus (born 16 October 1987) is a New Zealand professional tennis player.

 He reached a career high singles ranking of 274 in July 2011, but is much better known as a leading doubles player with a huge serve.

 His highest ranking in this discipline is 12, gained in July 2017.

 He won the final of the 2017 French Open partnering with American Ryan Harrison, and followed that with a runner-up finish in the 2017 US Open mixed doubles
with Chan Hao-ching of Chinese Taipei as his partner.
</item 3>

<item 4>
2001 Milan Indoor – Singles
Marc Rosset was the defending champion at the Milan Indoor tennis tournament but lost in the second round to Greg Rusedski.
</item 4>

<item 5>
Martin Legner
Martin Legner (born December 17, 1961 in Tyrol) is a professional Austrian wheelchair tennis player who has been ranked number one for wheelchair doubles.

 Legner has won the Australian Open doubles title four times with Robin Ammerlaan since 2000, and has won eight doubles titles.

 Six of them are Australian Open Titles, the other two being French Open titles.

 He has won one French Open and one Australian Open singles title.

 He has represented his country at every Summer Paralympics since 1992, and has competed in both singles and doubles at all of those Games.

 His favorite surfaces are clay and hard court.

 As of April 7, 2007 his highest singles rank was number three.

 Also of that date, he was ranked number six for singles and doubles.

 His wheelchair is manufactured by Kronbickler.

 Legner is a fan of Bob Marley, and listens to his music aside from tennis.
</item 5>

<item 6>
Jürgen Melzer
Jürgen Melzer (born 22 May 1981 in Vienna) is an Austrian tennis player.

 He reached a career-high singles ranking of world No. 8 in April 2011, and a doubles ranking of world No. 6 in September 2010.

 He is a left-handed tennis player, but is right-handed in everyday life.

 He has a younger brother, Gerald Melzer, with whom he has played doubles in several tournaments.
</item 6>

<item 7>
2010 Australian Open – Mixed Doubles
Sania Mirza and Mahesh Bhupathi were the defending champions.

 They accepted the invitation for the tournament, but withdrew before their match against Carly Gullickson and Bernard Tomic and were replaced by Akgul
Amanmuradova and Rik de Voest.
</item 7>

<item 8>
Marc Rosset
Marc Rosset (born 7 November 1970) is a former professional tennis player from Switzerland who is best remembered for winning the men's singles gold medal at
the 1992 Olympic Games.

 He also won one Grand Slam doubles title at the French Open in 1992 partnering compatriot Jakob Hlasek.
</item 8>

<item 9>
Max Mirnyi
Maksim "Max" Mirnyi (Belarusian: Максім Мікалаевіч Мірны ] ) (Russian: Максим Николаевич Мирный ] ) (born 6 July 1977, in Minsk) is a Belarusian professional
tennis player.
</item 9>

<item 10>
2004 Canada Masters – Doubles
The 2004 Canada Masters – Doubles was the men's singles event of the one hundred and fifteenth edition of the Canada Masters; a WTA Tier I tournament and the
most prestigious men's tennis tournament held in Canada.

 Mahesh Bhupathi and Max Mirnyi were the defending champions.

 They were both present but did not compete together.

 Mirnyi partnered with Jonas Björkman, but Bhupathi and partner Leander Paes defeated them 6–4, 6–2, in the final.
</item 10>
=== END ITEM id=e64f7610e1be1d90 ===

=== ITEM id=8397068a691e32b3 ===
Gold sentences and facts:
Sentence 0: Pulandian District () is one of the seven districts under the administration of Dalian, located in the south of Liaoning province, People's Republic of China.
  Fact 0.0: Pulandian District is one of the seven districts under the administration of Dalian.
  Fact 0.1: Pulandian District is located in the south of Liaoning province, People's Republic of China.
Sentence 1: Hailar District (Mongolian: ᠬᠠᠶᠢᠯᠠᠷ ᠲᠣᠭᠣᠷᠢᠭ "Qayilar toɣoriɣ" Хайлаар тойрог, Cyrillic: Хайлаар дүүрэг, "Khailaar düüreg"; ) is an urban district that serves as the seat of the prefecture-level city Hulunbuir in northeastern Inner Mongolia, China.
  Fact 1.0: Hailar District's Mongolian name is ᠬᠠᠶᠢᠯᠠᠷ ᠲᠣᠭᠣᠷᠢᠭ, romanized as Qayilar toɣoriɣ.
  Fact 1.1: Hailar District's Mongolian name in Cyrillic script is Хайлаар тойрог.
  Fact 1.2: Hailar District's Cyrillic name is Хайлаар дүүрэг, romanized as Khailaar düüreg.
  Fact 1.3: Hailar District is an urban district.
  Fact 1.4: Hailar District serves as the seat of the prefecture-level city Hulunbuir.
  Fact 1.5: Hailar District is located in northeastern Inner Mongolia, China.

Retrieved items:
<item 1>
Qinghe District, Tieling
Qinghe District () is a district of Tieling, Liaoning province, People's Republic of China.

 However, it is located closer to Kaiyuan City than it is to downtown Tieling, which lies 39 km to the southwest.
</item 1>

<item 2>
Ruili
Ruili (; Burmese: ရွှေလီ ; Thai: เมืองมาว ) is a county-level city of Dehong Prefecture, in the west of Yunnan province, People's Republic of China.

 It is a major border crossing between China and Myanmar, with the town of Muse located across the border.
</item 2>

<item 3>
Shuangta, Pulandian
Shuangta () is a town under the administration of Pulandian City in southern Liaoning province, China, located 58 km northeast of downtown Pulandian and about
120 km northeast of Dalian.

 , it has 9 villages under its administration.
</item 3>

<item 4>
Pulandian District
Pulandian District () is one of the seven districts under the administration of Dalian, located in the south of Liaoning province, People's Republic of China.

 Its area is 2769.90 km² and its permanent population as of 2010 is 741,230.

 The district borders the prefecture-level city of Yingkou to the north.
</item 4>

<item 5>
Lingdian
Zero Band () are a Chinese rock band from Inner Mongolia.

 The lead singer is Zhou Xiaoou (周晓鸥).

 The English name "Zero Band" was used in Beijing concert advertisements, though the band did not originally use an English name; the name has also been
translated as "Point Zero" in some older English publications.

 The band appeared on the Jingwen label in China, but were licensed outside China by JVC.

 One of the top Beijing bands at the end of the 1990s, Lingdian originally suffered the same resistance from government-owned venues as other indigenous PRC
rock bands.
</item 5>

<item 6>
Hulunbuir
Hulunbuir or Hulun Buir (Mongolian: ; , "Hūlúnbèi'ěr") is a region that is governed as a prefecture-level city in northeastern Inner Mongolia, in China.

 Its administrative center is located at Hailar District, its largest urban area.

 Major scenic features are the high steppes of the Hulun Buir grasslands, the Hulun and Buir lakes (the latter partially in Mongolia), and the Khingan range.

 Hulun Buir borders Russia to the north and west, Mongolia to the south and west, Heilongjiang province to the east and Hinggan League to the direct south.

 Hulunbuir is a linguistically diverse area: next to Mandarin Chinese, Mongolian dialects such as Khorchin and Buryat, the Mongolic language Dagur and some
Tungusic languages are spoken there.
</item 6>

<item 7>
Hailar District
Hailar District (Mongolian: ᠬᠠᠶᠢᠯᠠᠷ ᠲᠣᠭᠣᠷᠢᠭ "Qayilar toɣoriɣ" Хайлаар тойрог, Cyrillic: Хайлаар дүүрэг, "Khailaar düüreg"; ) is an urban district that serves as
the seat of the prefecture-level city Hulunbuir in northeastern Inner Mongolia, China.

 Hulunbuir, due to its massive size, is a city in administrative terms only, being mainly grassland and rural.

 Hailar, therefore, is a "de facto" city.

 Hailar can also refer to the urban area around the Hailar district, with Hulunbuir being the wider geographical region that contains the urban area.
</item 7>

<item 8>
Taiyang Subdistrict
Taiyang Subdistrict () is a subdistrict of Wafangdian, Dalian District, Liaoning, China.

 It is based around the small town of Taiyang, Taiyangsheng or Taiyangshengxiang, which lies 19 kilometres east by road of Fuzhou, 64 kilometres north by road
from Pulandian.

 To the east is the DFM reservoir.
</item 8>

<item 9>
Dujiangyan City
Dujiangyan () is a county-level city, a subdivision of Chengdu, the capital of Sichuan Province, People's Republic of China.

 Its north-west region forms a border with sourthern Ngawa Tibetan and Qiang Autonomous Prefecture.

 It has an area of 1208 km2 and had a population of 600,000 in 2003.
</item 9>

<item 10>
Old Barag Banner
Old Barag Banner (Mongolian: ᠬᠠᠭᠤᠴᠢᠨ ᠪᠠᠷᠭᠤ ᠬᠣᠰᠢᠭᠤ Қаучин Бару қосиу "Qaɣučin Barɣu qosiɣu"; ) is a banner of northeastern Inner Mongolia, People's Republic of
China.

 It is located 28 km from Hailar District, the administrative centre of Hulunbuir City, which administers this banner.
</item 10>
=== END ITEM id=8397068a691e32b3 ===

=== ITEM id=73af6f3e845de221 ===
Gold sentences and facts:
Sentence 0: Rick Ray is an American filmmaker best known for his 2006 documentary film 10 Questions for the Dalai Lama which he wrote, filmed and directed.
  Fact 0.0: Rick Ray is an American filmmaker.
  Fact 0.1: Rick Ray is best known for his 2006 documentary film 10 Questions for the Dalai Lama.
  Fact 0.2: Rick Ray wrote, filmed and directed 10 Questions for the Dalai Lama.
Sentence 1: John Alan Lasseter (born January 12, 1957) is an American animator, film director, screenwriter, and film producer.
  Fact 1.0: John Alan Lasseter was born January 12, 1957.
  Fact 1.1: John Alan Lasseter is an American animator.
  Fact 1.2: John Alan Lasseter is an American film director.
  Fact 1.3: John Alan Lasseter is an American screenwriter.
  Fact 1.4: John Alan Lasseter is an American film producer.
Sentence 2:  He is also the Principal Creative Advisor for Walt Disney Imagineering.
  Fact 2.0: John Alan Lasseter is also the Principal Creative Advisor for Walt Disney Imagineering.

Retrieved items:
<item 1>
Rick Carter
Rick Carter (born 1950) is an American production designer and art director.

 He is known for his work in the film "Forrest Gump", which earned him an Oscar nomination, as well as numerous nominations of other awards for his work in
"Amistad" and "A.I. Artificial Intelligence".

 Other films include "Cast Away", "War of the Worlds", "What Lies Beneath", "Jurassic Park", "Avatar", and "Back to the Future Part II" and "Part III".

 Many of the films that he has worked on are directed by Steven Spielberg or Robert Zemeckis.

 For his part in the Art Direction of "Avatar", he was awarded the Academy Award for Best Production Design alongside Robert Stromberg and Kim Sinclair.

 In 2013, Carter won his second Academy Award, for production design on Steven Spielberg's biopic, "Lincoln".
</item 1>

<item 2>
Red's Dream
Red's Dream is a 1987 American computer-animated short film produced by Pixar and directed by John Lasseter.

 The short film, which runs four minutes, stars Red, a unicycle.

 Propped up in the corner of a bicycle store on a rainy night, Red dreams about a better place.

 "Red's Dream" was Pixar's second computer-animated short following "Luxo Jr." in 1986, also directed by Lasseter.
</item 2>

<item 3>
John Lasseter
John Alan Lasseter (born January 12, 1957) is an American animator, film director, screenwriter, and film producer.

 He currently is the chief creative officer of Pixar Animation Studios, Walt Disney Animation Studios, and DisneyToon Studios.

 He is also the Principal Creative Advisor for Walt Disney Imagineering.
</item 3>

<item 4>
Toy Story 3
Toy Story 3 is a 2010 American 3D computer-animated comedy-drama film produced by Pixar Animation Studios for Walt Disney Pictures.

 It is the third installment in Pixar's "Toy Story" series, and the sequel to 1999's "Toy Story 2".

 It was directed by Lee Unkrich, the editor of the first two films and the co-director of "Toy Story 2", written by Michael Arndt, while Unkrich wrote the story
along with John Lasseter and Andrew Stanton, respectively director and co-writer of the first two films.

 The plot focuses on the toys Woody, Buzz Lightyear, and their friends dealing with an uncertain future as their owner, Andy, prepares to leave for college.
</item 4>

<item 5>
Lasseter Family Winery
Lasseter Family Winery is a winery located in Glen Ellen, Sonoma County, California.

 The winery was founded in 2000 by Pixar and Walt Disney Animation Studios CCO John Lasseter and his wife, Nancy Lasseter.

 The winery, once inhabited by the Grand Cru Winery, produces approximately 1,200 cases of French red wine blends annually, with the capacity to produce up to
6,000.

 The winery grows Bordeaux and Rhône varietals on 27 acres.

 One of the Lasseters' winemaking mentors was Jess Jackson, of Kendall-Jackson.
</item 5>

<item 6>
Luxo Jr.
Luxo Jr. is a 1986 American computer-animated short film produced by Pixar and directed by John Lasseter.

 The two-minute short film revolves around one larger and one smaller desk lamp.

 The larger lamp, named Luxo Sr., looks on while the smaller, "younger" Luxo Jr.

 plays exuberantly with a ball that it accidentally deflates.

 "Luxo Jr." was Pixar's first animation after Ed Catmull and John Lasseter left Industrial Light and Magic's computer division.

 It is the source of the hopping desk lamp included in Pixar's corporate logo.
</item 6>

<item 7>
Wreck-It Ralph
Wreck-It Ralph is a 2012 American 3D computer-animated fantasy-comedy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.

 It is the 52nd Disney animated feature film.

 The film was directed by Rich Moore, who has directed episodes of "The Simpsons" and "Futurama", and the screenplay was written by Phil Johnston and Jennifer
Lee from a story by Moore, Johnston, and Jim Reardon.

 John Lasseter served as the executive producer.

 The film features the voices of John C. Reilly, Sarah Silverman, Jack McBrayer, and Jane Lynch.

 The film tells the story of the eponymous arcade game villain who rebels against his role and dreams of becoming a hero.

 He travels between games in the arcade and ultimately must eliminate a dire threat that could affect the entire arcade and one that Ralph himself
unintentionally started.
</item 7>

<item 8>
Walt Disney Pictures
Walt Disney Pictures, Inc. is an American film production company and a subsidiary of Walt Disney Studios, owned by The Walt Disney Company.

 The division is based at the Walt Disney Studios in Burbank, California, and is the main producer of live-action feature films within the Walt Disney Studios
unit.

 It took on its current name in 1983.

 Today, in conjunction with the other units of Walt Disney Studios, Walt Disney Pictures is classified as one of Hollywood's "Big Six" film studios.

 Films produced by Walt Disney Animation Studios and Pixar Animation Studios are also released under this brand.
</item 8>

<item 9>
Rick Ray
Rick Ray is an American filmmaker best known for his 2006 documentary film 10 Questions for the Dalai Lama which he wrote, filmed and directed.

 His documentary "Lynching Charlie Lynch", about the trials of former medical marijuana dispensary owner Charles C. Lynch, premiered at the 2011 San Luis Obispo
International Film Festival on March 9.
</item 9>

<item 10>
Toy Story
Toy Story is a 1995 American computer-animated buddy comedy adventure film produced by Pixar Animation Studios for Walt Disney Pictures.

 The directorial debut of John Lasseter, "Toy Story" was the first feature-length computer-animated film and the first feature film produced by Pixar.

 Taking place in a world where anthropomorphic toys pretend to be lifeless whenever humans are present, the film's plot focuses on the relationship between
Woody, an old-fashioned pullstring cowboy doll (voiced by Tom Hanks), and Buzz Lightyear, an astronaut action figure (voiced by Tim Allen), as they evolve from
rivals competing for the affections of Andy, their owner, to friends who work together to be reunited with Andy as his family prepares to move to a new home.

 The screenplay was written by Joss Whedon, Andrew Stanton, Joel Cohen and Alec Sokolow, based on a story by Lasseter, Pete Docter, Stanton and Joe Ranft.

 The film features music by Randy Newman, and was executive-produced by Steve Jobs and Edwin Catmull.
</item 10>
=== END ITEM id=73af6f3e845de221 ===

=== ITEM id=3061b6894d385d57 ===
Gold sentences and facts:
Sentence 0:  She is married to Scott Stapp, former frontman of the band Creed, and current frontman for the band Art of Anarchy.
  Fact 0.0: The woman described is married to Scott Stapp.
  Fact 0.1: Scott Stapp is a former frontman of the band Creed.
  Fact 0.2: Scott Stapp is the current frontman for the band Art of Anarchy.
Sentence 1:  The band's third album, "Weathered", was released in 2001 with Tremonti handling bass before the band disbanded in 2004 due to increasing tension between members.
  Fact 1.0: The band's third album, “Weathered”, was released in 2001.
  Fact 1.1: Tremonti handled bass when “Weathered” was released.
  Fact 1.2: The band disbanded in 2004.
  Fact 1.3: The band disbanded in 2004 due to increasing tension between members.

Retrieved items:
<item 1>
Jaclyn Stapp
Jaclyn Nesheiwat Stapp (born July 29, 1980) is a beauty queen, philanthropist and fashion model with pageant roots in Florida and New York.

 She is married to Scott Stapp, former frontman of the band Creed, and current frontman for the band Art of Anarchy.

 Her most notable titles include Mrs. Florida America 2008 and Miss New York USA 2004.

 She is executive director of The Scott Stapp With Arms Wide Open Foundation.
</item 1>

<item 2>
Creedence Clearwater Revisited
Creedence Clearwater Revisited is an American rock band formed in 1995 by two former members of Creedence Clearwater Revival – a much more well-known band with
a very similar name.

 It is known for playing the music that was originally made famous by Creedence Clearwater Revival.

 The two common band members are Stu Cook (bass) and Doug "Cosmo" Clifford (drums).
</item 2>

<item 3>
Scott Weiland
Scott Richard Weiland ( ; born Scott Richard Kline; October 27, 1967 – December 3, 2015) was an American musician, singer and songwriter.

 During a career spanning three decades, Weiland was best known as the lead singer of the band Stone Temple Pilots from 1989 to 2002 and 2008 to 2013.

 He was also a member of supergroup Velvet Revolver from 2003 to 2008 and recorded one album with another supergroup, Art of Anarchy.

 He established himself as a solo artist as well, releasing three studio albums, two cover albums, and collaborations with several other musicians throughout
his career.
</item 3>

<item 4>
Creedence Clearwater Revival Covers the Classics
Creedence Clearwater Revival Covers the Classics is a compilation album by Creedence Clearwater Revival.

 Released in 2009, the album contains cover versions of songs as recorded by the band.
</item 4>

<item 5>
The Studdogs
The band was signed by Orange Recordings in Los Angeles and toured all over the country, sharing bills with the Demolition Doll Rods, Bob Log III, The
Fleshtones, and Immortal Lee County Killers and a one-time show with the Suicide Girls Burlesque Tour in Orlando.

 While in the Studdogs, Rich Evans began promoting concerts under the tag "Mutiny Productions".

 He now heads the independent "Florida's Dying" label, promoting and releasing material from the Florida-underground music scene.

 Kyle Justin joined the band on drums in early 2002 to replace Jason, who was moving to Texas.

 Kyle vacated his position as drummer in 2003 to pursue a music career in Los Angeles.

 Eric Gebhardt left the band in 2005 to pursue a solo career under the alias Red Mouth.
</item 5>

<item 6>
Doug Clifford
Douglas "Cosmo" Clifford (born April 24, 1945 in Palo Alto, California) is an American drummer, best known as a founding member of Creedence Clearwater Revival.

 After the group dissolved in the early 1970s, Clifford released a solo album and later joined CCR bassist Stu Cook in the Don Harrison Band.

 In 1995, Clifford and Cook formed the band Creedence Clearwater Revisited, performing live versions of Creedence Clearwater Revival songs.
</item 6>

<item 7>
Creed (band)
Creed is an American rock band formed in 1993 in Tallahassee, Florida.

 The band's best-known line-up consists of lead vocalist Scott Stapp, guitarist and vocalist Mark Tremonti, bassist Brian Marshall, and drummer Scott Phillips.

 Creed released two studio albums, "My Own Prison" in 1997 and "Human Clay" in 1999, before Marshall left the band in 2000.

 The band's third album, "Weathered", was released in 2001 with Tremonti handling bass before the band disbanded in 2004 due to increasing tension between
members.

 Tremonti, Marshall, and Phillips went on to found Alter Bridge while Stapp followed a solo career.
</item 7>

<item 8>
Creedence Clearwater Revival (album)
Creedence Clearwater Revival is the debut studio album by the American rock band Creedence Clearwater Revival, released in 1968.
</item 8>

<item 9>
Zachary Throne
As a singer, Zachary has sung on many TV and radio jingles.

 He provided the singing voice for the character, "Mark Winkle" on the television series, "California Dreams" as well as the singing voice for the character,
"Greg Brady" in the films, "The Brady Bunch Movie" and "A Very Brady Sequel".

 Currently, he resides in Las Vegas, Nevada where he performs in many shows.

 From 2012-2014, he was the lead singer/lead guitar player/bass player for the Sin City Sinners, a group that also featured former Faster Pussycat guitarist and
co-founder Brent Muscat as well as Slash bass player, Todd Kerns.

 .

 With the group, he recorded two albums, "DIVEBAR Days Revisited" and "A Sinners Christmas 2", both released in 2013.
</item 9>

<item 10>
Creedence Clearwater Revival: Box Set
Creedence Clearwater Revival: Box Set is a box set by Creedence Clearwater Revival, released in 2001.

 It contains all of their complete studio albums, two complete live albums, and material recorded by the band under their previous names "The Golliwogs" and
"The Blue Velvets".

 In November 2013 the box set was reissued with different artwork.

 This box set spans their career, and contains all of their complete materials.
</item 10>
=== END ITEM id=3061b6894d385d57 ===

=== ITEM id=3dfd8dc01d8da11d ===
Gold sentences and facts:
Sentence 0:  The area along Northern Boulevard is well known for its high-end premium open-air shopping center, the Americana Manhasset.
  Fact 0.0: The area along Northern Boulevard is well known for its high-end premium open-air shopping center.
  Fact 0.1: The high-end premium open-air shopping center is named the Americana Manhasset.
Sentence 1: International fashion designer labels and luxury goods companies—Hermes, Dior, Cartier, Bottega Veneta, Chanel, Fendi, Gucci, Louis Vuitton, MaxMara, Celine, Tiffany & Co., and Prada—maintain a total of more than 60 retail stores and restaurants on a 220000 sqft site surrounded by gardens designed by Oheme van Sweden.
  Fact 1.0: Hermes, Dior, Cartier, Bottega Veneta, Chanel, Fendi, Gucci, Louis Vuitton, MaxMara, Celine, Tiffany & Co., and Prada are international fashion designer labels and luxury goods companies.
  Fact 1.1: These companies maintain a total of more than 60 retail stores and restaurants.
  Fact 1.2: These stores and restaurants are on a 220000 sqft site.
  Fact 1.3: The 220000 sqft site is surrounded by gardens.
  Fact 1.4: The gardens surrounding the site were designed by Oheme van Sweden.

Retrieved items:
<item 1>
Christian Dior SE
Christian Dior SE (] ), commonly known as Dior, is a French luxury goods company controlled and chaired by French businessman Bernard Arnault, who also heads
LVMH – the world's largest luxury group.

 Dior itself holds 42.36% shares of and 59.01% voting rights within LVMH.
</item 1>

<item 2>
Museum of Jurassic Technology
The Museum of Jurassic Technology is a museum located at 9341 Venice Boulevard in the Palms district of Los Angeles, California.

 It was founded by David Hildebrand Wilson and Diana Drake Wilson (husband and wife) in 1988.
</item 2>

<item 3>
Estée Lauder Companies
The Estée Lauder Companies Inc. ( ) is an American manufacturer and marketer of prestige skincare, makeup, fragrance and hair care products.

 The company owns a diverse portfolio of brands, distributed internationally through retail channels and digital commerce, and has its headquarters in Midtown
Manhattan, New York City.
</item 3>

<item 4>
Americana Manhasset
International fashion designer labels and luxury goods companies—Hermes, Dior, Cartier, Bottega Veneta, Chanel, Fendi, Gucci, Louis Vuitton, MaxMara, Celine,
Tiffany & Co., and Prada—maintain a total of more than 60 retail stores and restaurants on a 220000 sqft site surrounded by gardens designed by Oheme van
Sweden.

 Peter Marino is Americana Manhasset's master architect.
</item 4>

<item 5>
Concord watch
Concord Watch Company is a Swiss luxury goods company that is part of the Movado group that owns Movado, Ebel, ESQ, Coach and Hugo Boss.

 Founded in 1908, Concord was purchased in 1970 by the North American Watch Company, which also distributed the Piaget and Corum lines of watches.

 From the late 1970s to the late 1980s, Concord produced what would become some of the most marketed and respected luxury quartz watches on the market.

 Concord watches became recognized in large cities as status symbols and were noted for their innovation and design.

 Flagship quartz models such as the Concord Centurion and Concord Delirium ranged from $2,000 to $20,000 surpassing the price of base automatic Rolex, Cartier
and Omega wristwatches.

 By the 1990s Concord watches fell out of style and grew obsolete amidst the re-branding of the company.

 While the brand still exists today, it never regained its market position nor visibility that it once had.
</item 5>

<item 6>
Mouawad
Mouawad is a privately held Swiss and Emirati luxury goods company that makes fashion accessories, jewelry and retail company that sources diamonds and
gemstones, designs, manufacturers, and sells jewelry collections, objects of art, and luxury watches.

 The firm has headquarters in Geneva, Switzerland, with a Middle East headquarters at Jumeirah Lakes Towers in Dubai.

 Founded in 1891 in Beirut, Lebanon by David Mouawad, the firm is now led by 4th generation co-guardians Fred Mouawad, Alain Mouawad and Pascal Mouawad.

 The firm's jewelry and watch design and manufacturing divisions are located in Switzerland and Asia Pacific.
</item 6>

<item 7>
Miracle Mile (Manhasset)
The Miracle Mile is a prominent shopping district in Manhasset on the North Shore of Long Island in Nassau County, New York.

 The area along Northern Boulevard is well known for its high-end premium open-air shopping center, the Americana Manhasset.
</item 7>

<item 8>
Bumble and bumble
Bumble and bumble Products, LLC is one of 27 brands owned by Estée Lauder Companies, Inc.

 It was founded as a New York City hair salon in 1977 by hairdresser and entrepreneur Michael Gordon.

 The company operates two flagship salons in Manhattan: a midtown location on East 56th Street and on the top two floors of The House of Bumble in the
Meatpacking District.
</item 8>

<item 9>
New Roc City
New Roc City, also known as New Rochelle Center, is an entertainment, retail and residential complex in the Downtown section of the city of New Rochelle in
Westchester County, New York.

 It is located at 33 LeCount Place, between Main Street North and Main Street South.

 The center was built on the site of the former New Rochelle Mall which closed in 1992.
</item 9>

<item 10>
Luxify
Luxify is a Hong Kong-based company, providing an online marketplace for new, vintage and pre-owned luxury goods.

 The company manages an online marketplace website in which members buy and sell a variety of luxury goods.

 It was Hong Kong's first online marketplace to buy and sell luxury goods The site has over 15 categories, including real estate, watches, jewelry, luxury cars,
yachts, aircraft, luxury handbags, art, collectibles, furniture, antiques and fine wines & spirits.

 As of March 2016, the website had over 22,000 listings from luxury dealers from all over the world, making a US$18.6 billion marketplace.

 Luxify also has offices in Singapore and London and local partners in Shanghai and Jakarta.
</item 10>
=== END ITEM id=3dfd8dc01d8da11d ===

=== ITEM id=dfaf90b9eef3b445 ===
Gold sentences and facts:
Sentence 0: Marcel Émile Verdet (13 March 1824 – 3 June 1866) was a French physicist.
  Fact 0.0: Marcel Émile Verdet was born on 13 March 1824.
  Fact 0.1: Marcel Émile Verdet died on 3 June 1866.
  Fact 0.2: Marcel Émile Verdet was a French physicist.
Sentence 1:  Verdet did much to champion the early theory of the conservation of energy in France through his editorial supervision of the "Annales de chimie et de physique".
  Fact 1.0: Marcel Émile Verdet did much to champion the early theory of the conservation of energy in France.
  Fact 1.1: Marcel Émile Verdet championed this theory in France through his editorial supervision of the 'Annales de chimie et de physique'.
Sentence 2: Annales de chimie et de physique (French for "Annals of Chemistry and of Physics") is a scientific journal that was founded in Paris, France, in 1789 under the title "Annales de chimie".
  Fact 2.0: Annales de chimie et de physique is French for 'Annals of Chemistry and of Physics'.
  Fact 2.1: Annales de chimie et de physique is a scientific journal.
  Fact 2.2: Annales de chimie et de physique was founded in Paris, France, in 1789.
  Fact 2.3: Annales de chimie et de physique was founded under the title 'Annales de chimie'.

Retrieved items:
<item 1>
Auguste Brancart
August or Auguste Brancart (21 July 1851 - 1894?)

 was a Belgian publisher of pornographic literature, credited with the first publication of "My Secret Life".

 He published translations of English pornography into French and vice versa for English publishers such as Edward Avery.

 He also published work of the Decadent movement such as "Monsieur Vénus" by Rachilde.
</item 1>

<item 2>
La Belle Assemblée
La Belle Assemblée (in full La Belle Assemblée or, Bell's Court and Fashionable Magazine Addressed Particularly to the Ladies) was a British women's magazine
published from 1806 to 1837, founded by John Bell (1745–1831).
</item 2>

<item 3>
Émile Verdet
Marcel Émile Verdet (13 March 1824 – 3 June 1866) was a French physicist.

 He worked in magnetism and optics, editing the works of Augustin-Jean Fresnel.

 Verdet did much to champion the early theory of the conservation of energy in France through his editorial supervision of the "Annales de chimie et de
physique".
</item 3>

<item 4>
Annales de chimie et de physique
Annales de chimie et de physique (French for "Annals of Chemistry and of Physics") is a scientific journal that was founded in Paris, France, in 1789 under the
title "Annales de chimie".

 One of the early editors was the French chemist Antoine Lavoisier.

 In 1815, it became the "Annales de chimie et de physique", and was published under that name for the next 100 years.
</item 4>

<item 5>
Index of Middle English Verse
The Index of Middle English Verse (IMEV) is a bibliographic index of poetry in Middle English.

 Its first print publication, in 1943, was an extension of Carleton Brown's "Register of Middle English Religious & Didactic Verse", augmented by the inclusion
of secular verse.

 This edition, edited by Brown and Rossell Hope Robbins, contained entries for over 4000 Middle English poems in more than 2000 manuscripts.

 In 1965 the index was supplemented by Robbins and John L. Cutler.

 The IMEV is available online through Virginia Polytechnic Institute and State University.
</item 5>

<item 6>
Guglielmo Verdirame
Guglielmo Verdirame (born in Reggio di Calabria, Italy) is a Professor of International Law at King's College London in the Department of War Studies and the
School of Law.

 He was previously a university lecturer in law at the University of Cambridge, Faculty of Law, and a Fellow of the Lauterpacht Centre for International Law.

 He practises as a barrister at 20 Essex Street Chambers.
</item 6>

<item 7>
The Henry James Review
The Henry James Review is a triannual peer-reviewed academic journal established in 1979 and is the official publication of the Henry James Society.

 It is dedicated to the scholarly, critical, and theoretical study of the American writer Henry James.

 Each issue focuses on a specific theme of interest and seeks to promote understanding and study of James' contributions.

 The journal is published by the Johns Hopkins University Press and the current editor-in-chief is Susan M. Griffin (University of Louisville).
</item 7>

<item 8>
The Picture of Dorian Gray
The Picture of Dorian Gray is a philosophical novel by Oscar Wilde, first published complete in the July 1890 issue of "Lippincott's Monthly Magazine".

 The magazine's editor feared the story was indecent, and without Wilde's knowledge, deleted roughly five hundred words before publication.

 Despite that censorship, "The Picture of Dorian Gray" offended the moral sensibilities of British book reviewers, some of whom said that Oscar Wilde merited
prosecution for violating the laws guarding the public morality.

 In response, Wilde aggressively defended his novel and art in correspondence with the British press, although he personally made excisions of some of the most
controversial material when revising and lengthening the story for book publication the following year.
</item 8>

<item 9>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
</item 9>

<item 10>
Entente Cordiale (opera)
Entente Cordiale is a comic opera in one act by Ethel Smyth with an English-language libretto by Smyth, who describes the work as "a post-war comedy in one act
(founded on fact)".

 It was first performed by students at the Royal College of Music in London on 22 July 1925.
</item 10>
=== END ITEM id=dfaf90b9eef3b445 ===

=== ITEM id=ba3f3ad863e1f50e ===
Gold sentences and facts:
Sentence 0:  The first of the books was optioned by Universal Pictures after a bidding war over the manuscript.
  Fact 0.0: The first of the books was optioned by Universal Pictures.
  Fact 0.1: The option followed a bidding war over the manuscript.
Sentence 1:  The company was founded in 1912 by Carl Laemmle, Mark Dintenfass, Charles O. Baumann, Adam Kessel, Pat Powers, William Swanson, David Horsley, Robert H. Cochrane, and Jules Brulatour, and is the oldest surviving film studio in the United States, the world's fourth oldest after Gaumont, Pathé and Nordisk Film, and the oldest in terms of the overall film market .
  Fact 1.0: The company was founded in 1912.
  Fact 1.1: The company was founded by Carl Laemmle, Mark Dintenfass, Charles O. Baumann, Adam Kessel, Pat Powers, William Swanson, David Horsley, Robert H. Cochrane, and Jules Brulatour.
  Fact 1.2: The company is the oldest surviving film studio in the United States.
  Fact 1.3: The company is the world's fourth oldest film studio, after Gaumont, Pathé and Nordisk Film.
  Fact 1.4: The company is the oldest film studio in terms of the overall film market.

Retrieved items:
<item 1>
John Dunning (film producer)
John Dunning (April 27, 1927–September 19, 2011) was a pioneering Canadian film producer from Montreal who co-founded the Canadian film production company
Cinépix and produced early works by notable Canadian directors David Cronenberg and Ivan Reitman.

 Dunning launched Cinépix with partner André Link in Montreal in the early 1960s.

 Their biggest commercial success—and the first Canadian box office hit—came with Reitman’s "Meatballs" (1979).
</item 1>

<item 2>
Simon Bent
Simon Bent is a British screenwriter and playwright, notable for work including BBC TV drama "" (2006), the screenplay for the feature film "Christie Malry's
Own Double-Entry" (2000), and the Joe Orton biographical play "Prick Up Your Ears" based on John Lahr's book.
</item 2>

<item 3>
Two Rivers Press
Two Rivers Press is an independent publishing house, based in the English town of Reading.

 Two Rivers Press was founded in 1994 by Peter Hay (1951–2003), a local artist.

 Its name reflects his enthusiasm for the town and its two rivers, the Kennet and the Thames, and its intention to explore "the place where art and history
meet".

 The name also gives a clue to the origins of the company in the, ultimately successful, opposition to Reading's proposed "Cross-Town Route", a road scheme that
would have seriously impacted the point at which the two rivers meet.
</item 3>

<item 4>
Bogle-L'Ouverture Publications
Bogle-L’Ouverture Publications (BLP) is a radical London-based publishing company founded in 1968 by Guyanese activists Jessica Huntley (23 February 1927 – 13
October 2013) and Eric Huntley (born 25 September 1929).

 Named in honour of two outstanding liberation fighters in Caribbean history, Toussaint L'Ouverture and Paul Bogle, the company began operating during a period
in the UK when "books by Black authors or written with a sympathetic view of Black people’s history and culture were rare in mainstream bookshops in the UK."

 Alongside New Beacon Books (founded in 1966) and Allison & Busby (founded in 1967), BLP was one of the first black-owned independent publishing companies in
the UK.

 BLP has been described as "a small, unorthodox, self-financing venture that brought a radical perspective to non-fiction, fiction, poetry and children's
books."
</item 4>

<item 5>
Field Enterprises
Field Enterprises was a private holding company founded on August 31, 1944, by Marshall Field III and others whose main asset was the "Chicago Sun".

 That same year the company acquired the book publishers Simon & Schuster and Pocket Books.
</item 5>

<item 6>
Some Kind of Fairy Tale
Some Kind of Fairy Tale is a 2012 novel by the British author Graham Joyce.

 A work of speculative fiction, it won the British Fantasy Society's Fantasy Novel of the Year award (the Robert Holdstock Award) in 2013.

 Film rights to the novel have been sold and a potential movie is in the development stage.
</item 6>

<item 7>
Simon Bloom
Simon Bloom is a series of young adult novels by Michael Reisman.

 Set in the fictional town of Lawnville, New Jersey, it is about a boy who finds a book that lets him control the laws of physics.

 The first of the books was optioned by Universal Pictures after a bidding war over the manuscript.
</item 7>

<item 8>
Applewood Books
Applewood Books is a book publishing company founded by Phil Zuckerman in 1976.

 They specialize in publishing exacting recreations of historic books, including complex reprints of children's art and pop-up books and other books published
by methods which duplicate antique publishing techniques.

 They have more than 2000 titles in print.

 In recent years, the company has been working to increase the number of reprints it publishes.

 In 2007, the company published over 300 titles.
</item 8>

<item 9>
Joshua Bloom
Joshua Simon Bloom (born June 8, 1974 in Washington, D.C.) is an American astrophysicist, full professor of astronomy at the University of California, Berkeley,
and was the CTO and co-founder of the machine-learning company wise.io (acquired by General Electric, 2016).

 He received a Bachelor of Arts in astronomy and astrophysics and physics from the Harvard College in 1996, an M.Phil from Cambridge University in 1997, and a
PhD in astronomy from the California Institute of Technology in 2002.

 He was a Junior Fellow of the Harvard Society of Fellows from 2002 to 2005.

 His astronomy research focuses on gamma-ray bursts and other astrophysical transients such as supernovae and tidal disruption events.

 He is author of the book "What Are Gamma-Ray Bursts?"

 published by Princeton University Press in 2011.
</item 9>

<item 10>
Simon &amp; Schuster
Simon & Schuster, Inc. ( ), a subsidiary of CBS Corporation, is an American publishing company founded in New York City in 1924 by Richard Simon and Max
Schuster.

 As of 2016, Simon & Schuster publishes 2,000 titles annually under 35 different imprints.
</item 10>
=== END ITEM id=ba3f3ad863e1f50e ===

=== ITEM id=cb44074ac526b247 ===
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
Flight and expulsion of Germans (1944–50)
During the later stages of World War II and the post-war period, German citizens and people of German ancestry were expelled from various Eastern European
countries and sent to the remaining territory of Germany and Austria.

 After 1950, some emigrated to the United States, Australia, and other countries from there.

 The areas affected included the former eastern territories of Germany, which were annexed by Poland and the Soviet Union after the war, as well as Germans who
were living within the prewar borders of Poland, Czechoslovakia, Hungary, Romania, Yugoslavia, and the Baltic States.

 The Nazis had made plans—only partially completed before the Nazi defeat—to remove many Slavic and Jewish people from Eastern Europe and settle the area with
Germans.

 The post war expulsion of the Germans formed a major part of the geopolitical and ethnic reconfiguration of Eastern Europe in the aftermath of World War II,
that attempted to create ethnically homogeneous nations within redefined borders.

 Between 1944 and 1948 about 31 million people, including ethnic Germans ('Volksdeutsche') as well as German citizens ('Reichsdeutsche'), were permanently or
temporarily moved from Central and Eastern Europe.
</item 1>

<item 2>
Soviet Unterzoegersdorf
Soviet Unterzoegersdorf (German: "Sowjet-Unterzögersdorf" ) is a fictitious country created by the art/technology/theory group monochrom.

 It is the "last existing appanage republic of the USSR", located inside the Republic of Austria.

 Unterzögersdorf is partially based on an existing village well-known to Johannes Grenzfurthner, the creator of the concept.

 As a kid and teenager, Grenzfurthner spent a lot of time at his grandparents' farm in the small village of Unterzögersdorf (a cadastral municipality of
Stockerau).

 His grandparents' stories about the Naziism, WWII and the Soviet occupation in allied-occupied Austria (1945-1955) form the inspirational basis of the
long-term project "Soviet Unterzoegersdorf".
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
Kross SA
Kross is a Polish company within the sports industry that makes bicycles and bicycle frames.

 The company is currently the largest in Poland manufacturing bicycles sold in the Polish market.

 In 2005 the company sold almost 800,000 products, half of which were exported to 35 countries, 30 of which are in Europe.
</item 7>

<item 8>
Knaster–Kuratowski fan
In topology, a branch of mathematics, the Knaster–Kuratowski fan (named after Polish mathematicians Bronisław Knaster and Kazimierz Kuratowski) is a specific
connected topological space with the property that the removal of a single point makes it totally disconnected.

 It is also known as Cantor's leaky tent or Cantor's teepee (after Georg Cantor), depending on the presence or absence of the apex.
</item 8>

<item 9>
Battle of Berestechko
The Battle of Berestechko (Polish: "Bitwa pod Beresteczkiem" ; Ukrainian: Берестецька битва, Битва під Берестечком ) was fought between the Ukrainian Cossacks,
led by Hetman Bohdan Khmelnytsky, aided by their Crimean Tatar allies, and a Polish army under King John II Casimir.

 It was a battle of a Cossack rebellion in Ukraine that took place in the years 1648–1657 after the expiration of a two-year truce.

 Fought from 28 to 30 June 1651, the battle took place in the province of Volhynia, on the hilly plain south of the Styr River.

 The Polish camp was on the river opposite Berestechko and faced south, towards the Cossack army about two kilometers away, whose right flank was against the
River Pliashivka (Pliashova) and the Tatar army on their left flank.

 It was probably the largest European land battle of the 17th century.
</item 9>

<item 10>
Kolejka (game)
Kolejka ( ; Polish for "queue" or "line (of people)") is a board game designed by Karol Madaj and launched in February 2011 by Poland's Institute of National
Remembrance.

 It was designed to teach young generations about the hardships under communism (1945–1989), particularly the difficulty of acquiring consumer goods in the
shortage economy.

 The game has been described as inspired by "Monopoly".
</item 10>
=== END ITEM id=cb44074ac526b247 ===
