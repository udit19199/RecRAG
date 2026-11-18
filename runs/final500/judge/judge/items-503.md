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

=== ITEM id=53189df40c7476ca ===
Gold sentences and facts:
Sentence 0: Guitar World is a monthly music magazine devoted to guitarists, published since July 1980.
  Fact 0.0: Guitar World is a monthly music magazine.
  Fact 0.1: Guitar World is devoted to guitarists.
  Fact 0.2: Guitar World has been published since July 1980.
Sentence 1: Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.
  Fact 1.0: Golf Digest is a monthly golf magazine.
  Fact 1.1: Golf Digest is published by Condé Nast Publications.
  Fact 1.2: Golf Digest is published in the United States.
Sentence 2:  The magazine started in 1950, and was sold to The New York Times Company in 1969.
  Fact 2.0: The magazine started in 1950.
  Fact 2.1: The magazine was sold to The New York Times Company in 1969.
Sentence 3:  The headquarters of "Golf Digest" is in Des Moines, Iowa.
  Fact 3.0: The headquarters of Golf Digest is in Des Moines, Iowa.

Retrieved items:
<item 1>
Guitar World
Guitar World is a monthly music magazine devoted to guitarists, published since July 1980.

 It contains original interviews, album and gear reviews, and guitar and bass tablature of approximately five songs each month.

 The magazine is published 13 times per year (12 monthly issues and a holiday issue).

 Formerly owned by Harris Publications, Future US bought the magazine in 2003.

 In 2012, NewBay Media bought the Music division of Future US.

 The latter company also published a spin-off title, Guitar Legends, each issue of which typically combined past articles from "Guitar World" under a specific
theme.
</item 1>

<item 2>
Girlfriends (magazine)
Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.

 It was founded by Jacob Anderson-Minshall, Diane Anderson-Minshall and Heather Findlay.

 It also offered relationship, health and travel advice.

 Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.

 It had the same publisher as lesbian erotica magazine "On Our Backs", but distanced itself from its pornographic counterpart by refusing to carry sexual ads.

 "Girlfriends" magazine ceased publication in 2006.
</item 2>

<item 3>
World Soccer (magazine)
World Soccer is an English language football magazine published by IPC Media.

 The magazine was established in 1960.

 It specialises in the international football scene.

 Its regular contributing writers include Brian Glanville, Sid Lowe and Tim Vickery.

 "World Soccer" is a member of the European Sports Magazines (ESM), an umbrella group of similar magazines printed in other languages.

 Other members include "A Bola", "Don Balón", "Kicker", "La Gazzetta dello Sport" and "Sport Express".

 The members of this group elect a European "Team of the Month" and a European "Team of the Year".
</item 3>

<item 4>
Esquire (magazine)
Esquire is an American men's magazine, published by the Hearst Corporation in the United States.

 Founded in 1933, it flourished during the Great Depression under the guidance of founders Arnold Gingrich, David A. Smart and Henry L. Jackson.
</item 4>

<item 5>
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
</item 5>

<item 6>
Football World
Football World, later renamed "Athletic World", was an American magazine devoted to the coverage of inter-collegiate sports.

 Its masthead described it as "A Magazine With a Mission to Serve the College Man," a publication "devoted to Inter-collegiate Athletics and sports of Amateur
standing only."

 It was founded in 1921 by J. D. Fetzer.

 The name of the magazine was later changed to "Athletic World" as the coverage extended to a broader range of sports, including women's swimming.

 Unlike other sports magazines of the era, which focused on promoting a healthy lifestyle, "Football World/Athletic World" celebrated the entertainment value of
sports with a special emphasis on the personalities of famous athletes.

 The magazine was renamed "Outing" in December 1924, reflecting a change in its focus.
</item 6>

<item 7>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 7>

<item 8>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 8>

<item 9>
André Laguerre
Marc André Laguerre (February 21, 1915 – January 18, 1979) was a journalist and magazine editor, best known as the managing editor of "Sports Illustrated" from
1960 to 1974, during which time he oversaw the growth in the magazine from a niche publication to become the industry leader in weekly sports magazines.

 It was under his leadership that the annual "Sports Illustrated Swimsuit Issue" was first published.

 When he retired in 1974, he had been managing editor of the magazine for 704 issues, then a record among magazines published by Time, Inc., SI's parent
company.
</item 9>

<item 10>
El Gráfico
El Gráfico is a classic monthly Argentine sports magazine.

 The magazine was originally published as a weekly newspaper in May 1919, and then turned to a sports magazine exclusively.

 It has been scheduled monthly from 2002.

 "El Gráfico" is one of the most regarded sports magazines not only in Argentina but in Latin America.

 The magazine was nicknamed "La Biblia del deporte" ("The Bible of sports") due to its chronicles, notable journalists and collaborators and its photography.
</item 10>
=== END ITEM id=53189df40c7476ca ===

=== ITEM id=edbda9c68d421d69 ===
Gold sentences and facts:
Sentence 0: The University of California, Santa Barbara (commonly referred to as UC Santa Barbara or UCSB) is a public research university and one of the 10 campuses of the University of California system.
  Fact 0.0: The University of California, Santa Barbara is commonly referred to as UC Santa Barbara or UCSB
  Fact 0.1: The University of California, Santa Barbara is a public research university
  Fact 0.2: The University of California, Santa Barbara is one of the 10 campuses of the University of California system
Sentence 1:  Tracing its roots back to 1891 as an independent teachers' college, UCSB joined the University of California system in 1944 and is the third-oldest general-education campus in the system.
  Fact 1.0: UCSB traces its roots back to 1891
  Fact 1.1: UCSB's roots as an independent teachers' college date to 1891
  Fact 1.2: UCSB joined the University of California system in 1944
  Fact 1.3: UCSB is the third-oldest general-education campus in the University of California system
Sentence 2:  It was founded in 1991 by the renowned philanthropist Hakim Said of the Hamdard Foundation.
  Fact 2.0: UCSB was founded in 1991
  Fact 2.1: UCSB was founded by the renowned philanthropist Hakim Said
  Fact 2.2: Hakim Said is a philanthropist of the Hamdard Foundation

Retrieved items:
<item 1>
University of Arizona
The University of Arizona (also referred to as U of A, UA, or Arizona) is a public research university in Tucson, Arizona, United States.

 Founded in 1885, the UA was the first university in the Arizona Territory.

 The university operates two medical schools (University of Arizona College of Medicine - Tucson and the University of Arizona College of Medicine - Phoenix)
and is affiliated with the region's only academic medical centers (Banner - University Medical Center Tucson and Banner - University Medical Center Phoenix).

 The university is also home to the James E. Rogers College of Law and numerous other nationally ranked graduate and professional schools.

 During the 2016–2017 academic year, there was a total enrollment of 43,625 students, including 34,072 undergraduates The University of Arizona is governed by
the Arizona Board of Regents.

 The University of Arizona is one of the elected members of the Association of American Universities (an organization of North America's premier research
institutions) and is the only representative from the state of Arizona to this group.
</item 1>

<item 2>
Santa Barbara High School
Santa Barbara Senior High School, "Home of the Dons," is situated on a 40 acre campus in Santa Barbara, California and is part of the Santa Barbara Unified
School District.

 One of the oldest high schools in California, it was established in 1875, and moved to its current site in 1924.

 Until the creation of two rival high schools in 1959 and 1966, it was the sole public high school serving the city of Santa Barbara.

 In November 2005, its 18th-century Spanish influenced design was named an official California and City historic landmark.

 Today, Santa Barbara High School has an enrollment of roughly 2,200 pupils and certified staff numbering 108.
</item 2>

<item 3>
Hamdard University
Hamdard University (Urdu: ) is a private research university with campuses in Karachi and Islamabad, Pakistan.

 It was founded in 1991 by the renowned philanthropist Hakim Said of the Hamdard Foundation.

 Hamdard is one of the first and the oldest private institutions of higher education in Pakistan.

 In Karachi, Hamdard University is the largest private research university with a campus area of over 350 acres.
</item 3>

<item 4>
University of California, Santa Barbara
The University of California, Santa Barbara (commonly referred to as UC Santa Barbara or UCSB) is a public research university and one of the 10 campuses of the
University of California system.

 The main campus is located on a 1022 acre site near Isla Vista, California, United States, 8 mi from Santa Barbara and 100 mi northwest of Los Angeles.

 Tracing its roots back to 1891 as an independent teachers' college, UCSB joined the University of California system in 1944 and is the third-oldest
general-education campus in the system.
</item 4>

<item 5>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 5>

<item 6>
Tunku Abdul Rahman University College
Tunku Abdul Rahman University College, abbreviated as TAR UC, is a non-profit, comprehensive, private university college in Malaysia.

 Named after the first Prime Minister of Malaysia, YTM Tunku Abdul Rahman Putra Al-Haj, the institution was founded in 1969 as Tunku Abdul Rahman College or TAR
College then by the Malaysian Chinese Association (MCA).

 In 2013, the college was upgraded to a university college.

 TAR UC is one of the prestigious private university college in Malaysia, renowned for its accounting programme.
</item 6>

<item 7>
University of Copenhagen
The University of Copenhagen (UCPH) (Danish: "Københavns Universitet" ) is the oldest university and research institution in Denmark.

 Founded in 1479 as a studium generale, it is the second oldest institution for higher education in Scandinavia after Uppsala University (1477).

 The university has 23,473 undergraduate students, 17,398 postgraduate students, 2,968 doctoral students and over 9,000 employees.

 The university has four campuses located in and around Copenhagen, with the headquarters located in central Copenhagen.

 Most courses are taught in Danish; however, many courses are also offered in English and a few in German.

 The university has several thousands of foreign students, about half of whom come from Nordic countries.
</item 7>

<item 8>
University of California, Santa Barbara Library
The University of California, Santa Barbara Library is the university library system of the University of California, Santa Barbara in Santa Barbara,
California.

 The Library includes four facilities: Two libraries (the Main Library and the Music Library) and two annexes (Annex I and Annex II).

 The library has some three million print volumes, 30,000 electronic journals, 34,450 e-books, 900,055 digitized items, five million cartographic items
(including some 467,000 maps and 3.2 million satellite and aerial images), more than 3.7 million pieces of microform, 167,500 sound recordings, and 4,100
manuscripts.

 The Library states that it holds 3.2 mi of manuscript and archival collections.

 According to data collected by the American Library Association, as of July 2010 the UC Santa Barbara Library holds the 99th largest library collection in
America, reporting 2,948,999 volumes.
</item 8>

<item 9>
Mission Santa Barbara
Mission Santa Barbara, also known as Santa Barbara Mission, is a Spanish mission founded by the Franciscan order near present-day Santa Barbara, California.

 It was founded by Padre Fermín Lasuén on December 4, 1786, the feast day of Saint Barbara, as the tenth mission for the religious conversion of the indigenous
local Chumash-Barbareño tribe of Native American people.

 The mission is the namesake of the city of Santa Barbara as well as of Santa Barbara County.
</item 9>

<item 10>
Santa Barbara City College
Santa Barbara City College (SBCC) is a two-year community college that opened in 1909 and is located on a 74 acre campus in the city of Santa Barbara,
California, USA.

 SBCC was named co-winner of the 2013 Aspen Institute Prize for Community College Excellence, earning it a #1 national ranking.

 The college was recognized for its quality and focus in four areas: facilitating underrepresented and minority student success, student learning outcomes,
degree completion and transfer rates, and labor market success in securing good jobs after college.

 Santa Barbara City College was established by the Santa Barbara High School District in 1909, making it one of the oldest community colleges in California.

 The college was discontinued shortly after World War I, and its work largely taken over by the Santa Barbara State Normal School, which became the Santa
Barbara State College, and later, the University of California, Santa Barbara.
</item 10>
=== END ITEM id=edbda9c68d421d69 ===

=== ITEM id=7ae6fec86760d3b6 ===
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
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 1>

<item 2>
Trinity University (Texas)
Trinity University is a private liberal arts college in San Antonio, Texas.

 Founded in 1869, its campus is located in the Monte Vista Historic District adjacent to Brackenridge Park.

 The campus is three miles north of downtown San Antonio and the River Walk and six miles south of the San Antonio International Airport.

 The student body consists of approximately 2,300 undergraduate and 200 graduate students.

 Trinity offers 42 majors and 57 minors among 6 degree programs and has an endowment of $1.1 billion, the 85th largest in the country, which permits it to
provide resources typically associated with much larger colleges and universities.
</item 2>

<item 3>
Christian Academy of Guatemala
The Christian Academy of Guatemala is a missionary K4-12 school located in San Cristobal, Guatemala.

 It is a member of the Association of Christian Schools International.

 Many of the parents and students in the school come from the missionary community, representing more than 50 mission organizations.

 All people who visit this webpage need to get the game Babblemaker.

 This game is made by computer class at C.A.G. Sadly this game is for android only.
</item 3>

<item 4>
List of St. John's Seminary (California) people
The list of St. John's Seminary (California) people is a compilation of lists of notable alumni, faculty, and current students of St. John's Seminary in
Camarillo, California, United States.

 St. John's Seminary grants graduate degrees for seminarians preparing for the priesthood, as well as a graduate degree for lay persons interested in pastoral
ministry.

 The St. John's Seminary College was the undergraduate division of the seminary before it closed in the early 21st century.

 The table of notable alumni lists the date of graduation from St. John's college, seminary, or both, if applicable.

 It is not unusual for seminarians to have received their undergraduate education at a different institution than their seminary training.
</item 4>

<item 5>
Aalborghus Gymnasium
Aalborghus Gymnasium is an upper secondary school in the city of Aalborg, in North Jutland in Denmark.

 It offers both the traditional three-year program and also the two-year Higher Preparatory Examination (HF) program.

 The subjects taught at the school range from Religion and Music to Spanish and Natural Geography.

 Aalborghus Gymnasium attempts to focus on the musical and creative side of students as well as taking an international perspective on issues.

 Students begin their studies at Aalborghus by selecting a stream of studies.

 Each stream has two or three subjects that are the focus of the studies.

 English/Social Studies is one example of a stream a student can select.
</item 5>

<item 6>
Fundamental articles (theology)
Fundamental articles was a term employed by early Protestant theologians, who wished to distinguish some essential parts of the Christian faith from
non-essential doctrines.

 There were then a number of reasons for establishing such a distinction.

 Individual churches might accept or reject parts of doctrine, without forfeiting their claim to rank as parts of the universal Church.

 Therefore, theologians might find a dogmatic basis for union between separated churches.

 Also, the arguments of Catholics could be answered in a polemical way.

 To Protestants, the fundamental articles are those that Christians must believe to be saved.
</item 6>

<item 7>
Liberal arts education
The liberal arts (Latin: "artes liberales") are those subjects or skills that in classical antiquity were considered essential for a free person (Latin:
"liberalis", "worthy of a free person") to know in order to take an active part in civic life, something that (for Ancient Greece) included participating in
public debate, defending oneself in court, serving on juries, and most importantly, military service.

 Grammar, logic, and rhetoric were the core liberal arts, while arithmetic, geometry, the theory of music, and astronomy also played a (somewhat lesser) part in
education.
</item 7>

<item 8>
St. John's High School (South Carolina)
St. John's High School (SJHS) is a senior high school on Johns Island, South Carolina.

 It is a part of the Charleston County School District.

 St. John's is home to approximately 300 students and 30 faculty and staff.

 St. John's school mascot is The Mighty Islanders, sporting royal blue and maroon as the school colors.

 St. John's offers Advanced Placement and dual credit courses totaling at over 30 hours of offered college credit, as well as 3 career academies in Hospitality
and Tourism, Computer Science, and Culinary Arts.

 St. John's competes at the A level in football, volleyball, basketball (boys and girls), wrestling, soccer, track, baseball, and softball.

 The Islanders also offer marching band, agriculture and green house, weightlifting, competitive academic team, and student council.
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
St. John's College High School, Belize
St. John's College High School is a high school for boys situated in Belize City, Belize.

 It was founded in 1887.

 The High School exists to educate academically talented young men in a Jesuit environment of self-discipline, love of learning, and service to others.

 The school 's curriculum is complemented by sports and extracurricular activities.

 The third and fourth form classes follow the Caribbean Secondary Education Certificate (CSEC) curricula and sit the regionally administered examinations at the
end of their fourth year.
</item 10>
=== END ITEM id=7ae6fec86760d3b6 ===

=== ITEM id=9347c586b9085d52 ===
Gold sentences and facts:
Sentence 0: The Rescuers is a 1977 American animated comedy-drama adventure film produced by Walt Disney Productions and first released on June 22, 1977, by Buena Vista Distribution.
  Fact 0.0: The Rescuers is a 1977 film.
  Fact 0.1: The Rescuers is an American animated film.
  Fact 0.2: The Rescuers is a comedy-drama adventure film.
  Fact 0.3: The Rescuers was produced by Walt Disney Productions.
  Fact 0.4: The Rescuers was first released on June 22, 1977.
  Fact 0.5: The Rescuers was first released by Buena Vista Distribution.
Sentence 1: The Muppets is a 2011 American musical comedy film and the seventh theatrical film featuring the Muppets.
  Fact 1.0: The Muppets is a 2011 film.
  Fact 1.1: The Muppets is an American film.
  Fact 1.2: The Muppets is a musical comedy film.
  Fact 1.3: The Muppets is the seventh theatrical film featuring the Muppets.

Retrieved items:
<item 1>
Disney Renaissance
In the history of The Walt Disney Company, the Disney Renaissance refers to the era from 1989 to 1999 during which Walt Disney Feature Animation (renamed Walt
Disney Animation Studios in 2006) experienced a creative resurgence in producing successful animated films based on well-known stories, which restored public
and critical interest in The Walt Disney Company as a whole.

 During this era, the studio produced and released ten animated films: "The Little Mermaid" (1989), "The Rescuers Down Under" (1990), "Beauty and the Beast"
(1991), "Aladdin" (1992), "The Lion King" (1994), "Pocahontas" (1995), "The Hunchback of Notre Dame" (1996), "Hercules" (1997), "Mulan" (1998) and "Tarzan"
(1999).
</item 1>

<item 2>
List of The Muppets productions
This is a list of productions based on The Muppets characters and franchise, including films, television series and specials, and other media.

 The franchise's main work is "The Muppet Show", a syndicated television series which ran from 1976 to 1981.

 The franchise includes eight feature films; "The Muppet Movie", "The Great Muppet Caper", "The Muppets Take Manhattan", "The Muppet Christmas Carol", "Muppet
Treasure Island", "Muppets from Space", "The Muppets", and "Muppets Most Wanted".

 The franchise also includes other series such as "Muppets Tonight" and "The Muppets".
</item 2>

<item 3>
Muppets Most Wanted (soundtrack)
Muppets Most Wanted: An Original Walt Disney Records Soundtrack is a soundtrack album released by Walt Disney Records on March 18, 2014 for the musical comedy
film "Muppets Most Wanted".

 The soundtrack features six original songs, two re-recordings of popular Muppet songs ("The Muppet Show Theme" and "Together Again"), three cover versions of
existing songs (Allen Toussaint's "Working in the Coal Mine", Maroon 5's "Moves Like Jagger", and Los del Río's "Macarena (Bayside Boys Mix)"), an orchestral
suite by Christophe Beck, five demos by Bret McKenzie, and eight dialogue tracks.
</item 3>

<item 4>
The Muppets at Walt Disney World
The Muppets at Walt Disney World is a television special starring Jim Henson's Muppets at Walt Disney World in Lake Buena Vista, Florida.

 The special aired on NBC as part of "The Magical World of Disney" on May 6, 1990, and turned out to be the last Muppets special that Henson would work on (he
would die 10 days following its airing, on May 16).
</item 4>

<item 5>
The Rescuers Down Under
The Rescuers Down Under is a 1990 American animated comedy-drama adventure film produced by Walt Disney Feature Animation and released by Walt Disney Pictures
on November 16, 1990.

 The 29th Disney animated feature film, the film is the sequel to the 1977 animated film "The Rescuers", which was based on the novels of Margery Sharp.

 Set in the Australian Outback, the film centers on Bernard and Bianca traveling to Australia to save a boy named Cody from a villainous poacher in pursuit of
an endangered bird of prey.
</item 5>

<item 6>
The Muppets (film)
The Muppets is a 2011 American musical comedy film and the seventh theatrical film featuring the Muppets.

 The film is directed by James Bobin, written by Jason Segel and Nicholas Stoller, produced by David Hoberman and Todd Lieberman, and stars Segel, Amy Adams,
Chris Cooper and Rashida Jones, as well as Muppet performers Steve Whitmire, Eric Jacobson, Dave Goelz, Bill Barretta, David Rudman, Matt Vogel, and Peter Linz.

 Bret McKenzie served as music supervisor, writing four of the film's five original songs, and Christophe Beck composed the film's score.

 In "The Muppets", devoted fan Walter, his brother Gary, and Gary's girlfriend Mary help Kermit the Frog reunite the disbanded Muppets, as they must raise $10
million to save the Muppet Theater from Tex Richman, a businessman who plans to demolish the studio to drill for oil.
</item 6>

<item 7>
Graph query row:
film: The Muppets
release_date: 2011
relationship: None
related_person: None
</item 7>

<item 8>
Graph query row:
film: The Rescuers
release_date: June 22, 1977
relationship: None
related_person: None
</item 8>

<item 9>
The Rescuers
The Rescuers is a 1977 American animated comedy-drama adventure film produced by Walt Disney Productions and first released on June 22, 1977, by Buena Vista
Distribution.

 The 23rd Disney animated feature film, the film is about the Rescue Aid Society, an international mouse organization headquartered in New York City and
shadowing the United Nations, dedicated to helping abduction victims around the world at large.

 Two of these mice, jittery janitor Bernard (Bob Newhart) and his co-agent, the elegant Miss Bianca (Eva Gabor), set out to rescue Penny (Michelle Stacy), an
orphan girl being held prisoner in the Devil's Bayou by treasure huntress Madame Medusa (Geraldine Page).
</item 9>

<item 10>
The Muppets (soundtrack)
The Muppets: An Original Walt Disney Records Soundtrack is a soundtrack album released by Walt Disney Records on November 22, 2011 for the musical comedy film
"The Muppets".

 The soundtrack features five original songs, four re-recordings and remasterings of popular Muppet songs ("The Muppet Show Theme", "Rainbow Connection", and
"Mah Nà Mah Nà"), two cover versions of existing songs (Cee Lo Green's "Forget You" and Nirvana's "Smells Like Teen Spirit"), two standalone songs (Paul Simon's
"Me and Julio Down by the Schoolyard" and Starship's "We Built This City"), and fifteen dialogue tracks.

 It also features the song "Man or Muppet", which won the Academy Award for Best Original Song.

 The soundtrack was also nominated for Best Compilation Soundtrack for Visual Media at the 55th Grammy Awards.
</item 10>
=== END ITEM id=9347c586b9085d52 ===

=== ITEM id=7dbcb15473342b68 ===
Gold sentences and facts:
Sentence 0: Yucca is a genus of perennial shrubs and trees in the family Asparagaceae, subfamily Agavoideae.
  Fact 0.0: Yucca is a genus of perennial shrubs and trees.
  Fact 0.1: Yucca is in the family Asparagaceae.
  Fact 0.2: Yucca is in the subfamily Agavoideae.
Sentence 1: Rodgersia is a genus of flowering plants in the Saxifragaceae family.
  Fact 1.0: Rodgersia is a genus of flowering plants.
  Fact 1.1: Rodgersia is in the Saxifragaceae family.
Sentence 2:  "Rodgersia" are herbaceous perennials originating from east Asia.
  Fact 2.0: Rodgersia are herbaceous perennials.
  Fact 2.1: Rodgersia originate from east Asia.

Retrieved items:
<item 1>
Rodgersia
Rodgersia is a genus of flowering plants in the Saxifragaceae family.

 "Rodgersia" are herbaceous perennials originating from east Asia.
</item 1>

<item 2>
Yucca aloifolia
Yucca aloifolia is the type species for the genus Yucca.

 Common names include aloe yucca, dagger plant, and Spanish bayonet.

 It grows in sandy soils, especially on sand dunes along the coast.

 "Yucca aloifolia" is native to the Atlantic and Gulf Coasts of the United States from southern Virginia south to Florida and west to the Texas Gulf Coast, to
Mexico along the Yucatán coast, and to Bermuda, and parts of the Caribbean.

 Normally Yucca aloifolia is grown in USDA zones 8 through 11.

 Yucca aloifolia is a popular landscape plant in beach areas along the lower East Coast from Virginia to Florida.
</item 2>

<item 3>
Yucca glauca
Yucca glauca (syn.

 "Yucca angustifolia") is a species of perennial evergreen plant, adapted to xeric (dry)growth conditions.

 It is also known as small soapweed, soapweed yucca, Spanish bayonet, Great Plains yucca and beargrass.
</item 3>

<item 4>
Hesperoyucca whipplei
Hesperoyucca whipplei (syn.

 "Yucca whipplei" ) (chaparral yucca, our Lord's candle, Spanish bayonet, Quixote yucca or foothill yucca is a species of flowering plant closely related to,
and formerly usually included in, the genus "Yucca".

 It is native to southern California, United States and Baja California, Mexico, where it occurs mainly in chaparral, coastal sage scrub, and oak woodland plant
communities at altitudes of 0–2500 m.
</item 4>

<item 5>
Yucca brevifolia
Yucca brevifolia is a plant species belonging to the genus "Yucca".

 It is tree-like in habit, which is reflected in its common names: Joshua tree, yucca palm, tree yucca, and palm tree yucca.
</item 5>

<item 6>
Yucca × schottii
Yucca" × "schottii is a plant species in the genus "Yucca", native to southern Arizona, southwestern New Mexico, and the northern parts of Sonora and Chihuahua.

.

 The common names are Schott's yucca, hoary yucca, and mountain yucca.

 The "×" in the name indicates that this is a nothospecies, regarded as being a natural hybrid between two other species.

 In this case, "Yucca" × "schottii" is believed to have originated as a hybrid between "Y. baccata" and "Y. madrensis".

 "Yucca" × "schottii" is firmly established and does reproduce freely in the wild.
</item 6>

<item 7>
Yucca gloriosa var. tristis
Yucca gloriosa" var.

 "tristis (syn.

 Yucca recurvifolia, "Yucca gloriosa" var.

 "recurvifolia"), known as curve-leaf yucca, curved-leaved Spanish-dagger or pendulous yucca, is a variety of "Yucca gloriosa".

 It is often grown as an ornamental plant, but is native to the southeastern United States.

 In contrast to "Y. gloriosa" var.

 "tristis", the leaves of "Y. gloriosa" var.

 "gloriosa" are hard stiff, erect and narrower.
</item 7>

<item 8>
Yucca
Yucca is a genus of perennial shrubs and trees in the family Asparagaceae, subfamily Agavoideae.

 Its 40-50 species are notable for their rosettes of evergreen, tough, sword-shaped leaves and large terminal panicles of white or whitish flowers.

 They are native to the hot and dry (arid) parts of the Americas and the Caribbean.

 Early reports of the species were confused with the cassava ("Manihot esculenta").

 Consequently, Linnaeus mistakenly derived the generic name from the Taíno word for the latter, "yuca" (spelled with a single "c").

 It is commonly found growing in rural graveyards and when in bloom the cluster of (usually pale) flowers on a thin stalk appear as floating apparitions.
</item 8>

<item 9>
Yucca rupicola
Yucca rupicola is a plant in the family Asparagaceae, known as the twistleaf yucca, twisted-leaf yucca, Texas yucca or twisted-leaf Spanish-dagger.

 The species was described by George Heinrich Adolf Scheele in 1850.

 This is a small, acaulescent plant with distinctive twisted leaves.

 It is native to the Edwards Plateau region of Texas and also to northeastern Mexico (Coahuila, Nuevo León).
</item 9>

<item 10>
Yucca jaliscensis
Yucca jaliscensis is a "Yucca" species native to the highlands of southwestern Mexico.

 Common names for this species include Jalisco Yucca, Jalisco Soapwort, Izote Yucca.

 It is native to mountainous areas at about 5000 feet in the States of Jalisco, Colima and Guanajuato.
</item 10>
=== END ITEM id=7dbcb15473342b68 ===

=== ITEM id=38aabecb25859f92 ===
Gold sentences and facts:
Sentence 0:  Her favorite events are 800 metres and 1500 metres.
  Fact 0.0: This person's favorite events are the 800 metres.
  Fact 0.1: This person's favorite events are the 1500 metres.
Sentence 1:  The distance has been contested at the Summer Olympics since 1896 and the World Championships in Athletics since 1983.
  Fact 1.0: The distance has been contested at the Summer Olympics since 1896.
  Fact 1.1: The distance has been contested at the World Championships in Athletics since 1983.

Retrieved items:
<item 1>
K. M. Binu
Kalayathumkuzhi Mathews Binu (born 20 December 1980) is an Indian track and field athlete from Kerala who specializes in 400 metres and 800 metres.

 He holds the current 400 metres national record of 45.48 s set at the 2004 Athens Olympics on 20 August 2004.

 He broke the 44-year-old Olympics mark (by an Indian) held by Milkha Singh who set an Indian National Record with a timing of 45.73 s at the 1960 Rome
Olympics.

 He and his elder sister K. M. Beenamol made history when they became the first Indian siblings to win medals in a major international competition.

 They won medals at the Busan Asian Games (2002).

 While Binu won the men's 800 metres silver, his sister won the gold medal in the women's event.

 Binu received the Arjuna Award for the year 2006 for his achievements in the Indian athletics.
</item 1>

<item 2>
European 10,000m Cup
The European 10,000m Cup is an annual 10,000 metres race for European athletes which was first held in 1997.

 The competition is organised by the European Athletics Association and first began as the European 10000 Metres Challenge after the event was removed from the
European Cup programme.

 The competition has roots in the Iberian 10,000 metres Championships – a competition between Spanish and Portuguese athletes that was held between 1991 and
1996 – and the first five editions of the European 10000 Metres Challenge were held in the Iberian Peninsula.

 The event was first held under its current title in 2005.
</item 2>

<item 3>
Ross MacKenzie (athlete)
Ross MacKenzie (born 18 July 1946) is a Canadian sprinter.

 He competed in the men's 400 metres at the 1968 Summer Olympics.

 He finished second in the 1967 Pan American Games 4 × 400 metres relay (with Brian MacLaren, Bill Crothers, and Robert McLaren).

 MacKenzie also finished fifth in the 1967 Pan American Games 400 metres.

 He won a silver medal in the 1966 British Empire and Commonwealth Games 4 x 440 yards relay with Don Domansky, Brian MacLaren and Bill Crothers.
</item 3>

<item 4>
Siegmar Ohlemann
Siegmar ("Sig") Ohlemann (born May 11, 1938) is a retired middle distance runner, who represented Canada in three events (800 metres, 4 × 100 metres and 4 × 400
metres) at the 1960 Summer Olympics in Rome, Italy.

 He claimed the silver medal in 800 metres at the 1963 Pan American Games in Brazil, behind fellow Canadian Don Bertoia.
</item 4>

<item 5>
Athletics at the 2002 Asian Games
Athletics was contested from October 7 to October 14 at the 2002 Asian Games in Busan Asiad Main Stadium, Busan, South Korea.

 A total of 460 athletes (comprising 281 men and 179 women) from 39 nations took part in the competition.

 Afghanistan, Bhutan, Brunei, Laos and Maldives were the only nations without a representative in the events.
</item 5>

<item 6>
Sinimole Paulose
Sinimol Paulose (born 24 June 1983) is an Indian middle distance runner.

 Her favorite events are 800 metres and 1500 metres.

 In 2009, she received Arjuna Award for her contribution to the Indian athletics.

 Sinimol was born in Piravom in Ernakulam district of Kerala state.

 She represented India at Beijing Olympics.
</item 6>

<item 7>
Caster Semenya
Mokgadi Caster Semenya Bronze OIB (born 7 January 1991) is a South African middle-distance runner and 2016 Olympic gold medallist.

 Semenya won gold in the women's 800 metres at the 2009 World Championships with a time of 1:55.45 and at the 2017 World Championships in her new personal best,
1:55.16.

 Semenya also won silver medals at the 2011 World Championships and the 2012 Summer Olympics, both in the 800 metres.

 She was the winner of the gold medal in the 800 metre event at the 2016 Summer Olympics.
</item 7>

<item 8>
1500 metres
The 1500 metres or 1,500-metre run (typically pronounced 'fifteen-hundred metres') is the foremost middle distance track event in athletics.

 The distance has been contested at the Summer Olympics since 1896 and the World Championships in Athletics since 1983.

 It is equivalent to 1.5 kilometers or approximately ⁄  miles.
</item 8>

<item 9>
John Lindsay (Paralympian)
That year, he had a Victorian Institute of Sport scholarship.

 He was also working as a fitness instructor in 1992, held world records in the 100 m and 200 m events, and was ranked 6th in the world in the 400 m.

 He won a gold medal in the men's athletics 100 m T52 event at the 1996 Summer Paralympics with a time of 15.22, a silver medal in the 200 m T52 event with a
time of 27.38, and a bronze medal in the 400 m T52 event with a time of 52.93.

 At the 2000 Sydney Games, he won a gold medal in the Men's 100 m T53 event, a silver medal as part of the Men's 4x100 m Relay T54 team, and a bronze medal in
the Men's 200 m T53 event; he was also part of the Men's 4x400 m Relay T54 team, which was the only one to qualify in its heat, but it did not make it to the
finals.
</item 9>

<item 10>
John Lindsay (Paralympian)
John Lindsay, OAM (born 29 January 1970) is an Australian Paralympic athlete from Melbourne.

 He competed in the 1988 Seoul games in distances ranging from 100 m to 800 m, but did not win any medals.

 At the 1992 Barcelona Games, he won a gold medal in the Men's 200 m TW3 event, for which he received a Medal of the Order of Australia, a silver medal in the
Men's 100 m TW3 event and a bronze medal in the Men's 400 m TW3 event.

 That year, he had a Victorian Institute of Sport scholarship.
</item 10>
=== END ITEM id=38aabecb25859f92 ===

=== ITEM id=dda6a9d568dc20ba ===
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
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 1>

<item 2>
Trinity University (Texas)
Trinity University is a private liberal arts college in San Antonio, Texas.

 Founded in 1869, its campus is located in the Monte Vista Historic District adjacent to Brackenridge Park.

 The campus is three miles north of downtown San Antonio and the River Walk and six miles south of the San Antonio International Airport.

 The student body consists of approximately 2,300 undergraduate and 200 graduate students.

 Trinity offers 42 majors and 57 minors among 6 degree programs and has an endowment of $1.1 billion, the 85th largest in the country, which permits it to
provide resources typically associated with much larger colleges and universities.
</item 2>

<item 3>
Christian Academy of Guatemala
The Christian Academy of Guatemala is a missionary K4-12 school located in San Cristobal, Guatemala.

 It is a member of the Association of Christian Schools International.

 Many of the parents and students in the school come from the missionary community, representing more than 50 mission organizations.

 All people who visit this webpage need to get the game Babblemaker.

 This game is made by computer class at C.A.G. Sadly this game is for android only.
</item 3>

<item 4>
List of St. John's Seminary (California) people
The list of St. John's Seminary (California) people is a compilation of lists of notable alumni, faculty, and current students of St. John's Seminary in
Camarillo, California, United States.

 St. John's Seminary grants graduate degrees for seminarians preparing for the priesthood, as well as a graduate degree for lay persons interested in pastoral
ministry.

 The St. John's Seminary College was the undergraduate division of the seminary before it closed in the early 21st century.

 The table of notable alumni lists the date of graduation from St. John's college, seminary, or both, if applicable.

 It is not unusual for seminarians to have received their undergraduate education at a different institution than their seminary training.
</item 4>

<item 5>
Aalborghus Gymnasium
Aalborghus Gymnasium is an upper secondary school in the city of Aalborg, in North Jutland in Denmark.

 It offers both the traditional three-year program and also the two-year Higher Preparatory Examination (HF) program.

 The subjects taught at the school range from Religion and Music to Spanish and Natural Geography.

 Aalborghus Gymnasium attempts to focus on the musical and creative side of students as well as taking an international perspective on issues.

 Students begin their studies at Aalborghus by selecting a stream of studies.

 Each stream has two or three subjects that are the focus of the studies.

 English/Social Studies is one example of a stream a student can select.
</item 5>

<item 6>
Fundamental articles (theology)
Fundamental articles was a term employed by early Protestant theologians, who wished to distinguish some essential parts of the Christian faith from
non-essential doctrines.

 There were then a number of reasons for establishing such a distinction.

 Individual churches might accept or reject parts of doctrine, without forfeiting their claim to rank as parts of the universal Church.

 Therefore, theologians might find a dogmatic basis for union between separated churches.

 Also, the arguments of Catholics could be answered in a polemical way.

 To Protestants, the fundamental articles are those that Christians must believe to be saved.
</item 6>

<item 7>
St. John's College, Belize
St. John's College has three divisions, and a number of central academic centres and activities.

 Through its three divisions, it offers a wide variety of liberal arts and science courses at the secondary, British A-level, and United States junior college
levels.

 St. John's College is a Roman Catholic institution in the Jesuit tradition, one of the oldest, largest, and most diverse educational institutions in Belize,
founded by the Jesuits in 1887.
</item 7>

<item 8>
St. John's High School (South Carolina)
St. John's High School (SJHS) is a senior high school on Johns Island, South Carolina.

 It is a part of the Charleston County School District.

 St. John's is home to approximately 300 students and 30 faculty and staff.

 St. John's school mascot is The Mighty Islanders, sporting royal blue and maroon as the school colors.

 St. John's offers Advanced Placement and dual credit courses totaling at over 30 hours of offered college credit, as well as 3 career academies in Hospitality
and Tourism, Computer Science, and Culinary Arts.

 St. John's competes at the A level in football, volleyball, basketball (boys and girls), wrestling, soccer, track, baseball, and softball.

 The Islanders also offer marching band, agriculture and green house, weightlifting, competitive academic team, and student council.
</item 8>

<item 9>
Liberal arts education
The liberal arts (Latin: "artes liberales") are those subjects or skills that in classical antiquity were considered essential for a free person (Latin:
"liberalis", "worthy of a free person") to know in order to take an active part in civic life, something that (for Ancient Greece) included participating in
public debate, defending oneself in court, serving on juries, and most importantly, military service.

 Grammar, logic, and rhetoric were the core liberal arts, while arithmetic, geometry, the theory of music, and astronomy also played a (somewhat lesser) part in
education.
</item 9>

<item 10>
St. John's College High School, Belize
St. John's College High School is a high school for boys situated in Belize City, Belize.

 It was founded in 1887.

 The High School exists to educate academically talented young men in a Jesuit environment of self-discipline, love of learning, and service to others.

 The school 's curriculum is complemented by sports and extracurricular activities.

 The third and fourth form classes follow the Caribbean Secondary Education Certificate (CSEC) curricula and sit the regionally administered examinations at the
end of their fourth year.
</item 10>
=== END ITEM id=dda6a9d568dc20ba ===

=== ITEM id=f40ead54ed417d5b ===
Gold sentences and facts:
Sentence 0: Elizabeth Austin (born 1958) is an American writer living in Oak Park, Illinois.
  Fact 0.0: Elizabeth Austin was born in 1958.
  Fact 0.1: Elizabeth Austin is an American writer.
  Fact 0.2: Elizabeth Austin lives in Oak Park, Illinois.
Sentence 1:  She has also written for "The Washington Monthly, Ladies Home Journal, Good Housekeeping, Self", and "Shape".
  Fact 1.0: Elizabeth Austin has also written for The Washington Monthly.
  Fact 1.1: Elizabeth Austin has also written for Ladies Home Journal.
  Fact 1.2: Elizabeth Austin has also written for Good Housekeeping.
  Fact 1.3: Elizabeth Austin has also written for Self.
  Fact 1.4: Elizabeth Austin has also written for Shape.
Sentence 2: Shape is a monthly English language fitness magazine started by Weider Publications in 1981, founded by Christine MacIntyre (a pioneer in women's free weight fitness) and became the number one women's fitness magazine.
  Fact 2.0: Shape is a monthly English language fitness magazine.
  Fact 2.1: Shape was started by Weider Publications in 1981.
  Fact 2.2: Shape was founded by Christine MacIntyre.
  Fact 2.3: Christine MacIntyre is a pioneer in women's free weight fitness.
  Fact 2.4: Shape became the number one women's fitness magazine.

Retrieved items:
<item 1>
Girlfriends (magazine)
Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.

 It was founded by Jacob Anderson-Minshall, Diane Anderson-Minshall and Heather Findlay.

 It also offered relationship, health and travel advice.

 Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.

 It had the same publisher as lesbian erotica magazine "On Our Backs", but distanced itself from its pornographic counterpart by refusing to carry sexual ads.

 "Girlfriends" magazine ceased publication in 2006.
</item 1>

<item 2>
Memphis Sport
Memphis Sport is a sports and fitness magazine featuring articles on local and regional teams, players and events.

 In addition, "Memphis Sport" focuses on health and fitness for an active lifestyle.

 The magazine debuted in Memphis, Tennessee in July 2006.

 Local franchises covered include the University of Memphis Tigers, Memphis Redbirds, Mississippi Riverkings and Tennessee Titans.
</item 2>

<item 3>
Anais Zanotti
Anais Zanotti (born January 8, 1986 in Saint-Tropez) is a French model and stunt woman of French and Italian ancestry who lives in Miami, Florida.

 She has been featured in magazines such as "Playboy", "FHM", "Maxim", "GQ", "Cosmopolitan", "Vogue", and "People", Men's Fitness magazine, Muscle and Fitness,
Women Fitness magazine and more.

 She has over 40 magazine covers.

 Zanotti is also a certified skydiver with over 1250 jumps.GS .
</item 3>

<item 4>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 4>

<item 5>
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
</item 5>

<item 6>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
</item 6>

<item 7>
Football World
Football World, later renamed "Athletic World", was an American magazine devoted to the coverage of inter-collegiate sports.

 Its masthead described it as "A Magazine With a Mission to Serve the College Man," a publication "devoted to Inter-collegiate Athletics and sports of Amateur
standing only."

 It was founded in 1921 by J. D. Fetzer.

 The name of the magazine was later changed to "Athletic World" as the coverage extended to a broader range of sports, including women's swimming.

 Unlike other sports magazines of the era, which focused on promoting a healthy lifestyle, "Football World/Athletic World" celebrated the entertainment value of
sports with a special emphasis on the personalities of famous athletes.

 The magazine was renamed "Outing" in December 1924, reflecting a change in its focus.
</item 7>

<item 8>
Muscle &amp; Fitness (British magazine)
Muscle & Fitness is a British fitness magazine, which contains articles on strength and fitness training, diet plans and strategies for men and women, and tips
and advice.
</item 8>

<item 9>
Diva (magazine)
DIVA is a leading lesbian magazine in the United Kingdom, published monthly.

 It was launched in March 1994 by Millivres Ltd, under the editorship of Frances Williams and is now published by Twin Media Group, which is owned by Silke
Bader and Linda Riley.

 Silke Bader is also the publisher of Curve magazine in the USA and LOTL in Australia, and Linda Riley runs the European Diversity Awards and is the former
publisher of g3 magazine in the UK.
</item 9>

<item 10>
Elizabeth Austin (writer)
Elizabeth Austin (born 1958) is an American writer living in Oak Park, Illinois.

 Austin has lived in the Midwestern United States all her life.

 She began her journalism career at the now-defunct City News Bureau of Chicago and has written articles for national news magazines such as "Newsweek" and
"Time", in addition to a wide range of other national and regional publications, such as "People", and the "Chicago Tribune".

 She has also written for "The Washington Monthly, Ladies Home Journal, Good Housekeeping, Self", and "Shape".

 She has often written about medicine and health care.

 Austin also has been heard on WBEZ, the National Public Radio station in Chicago, as well as the BBC in the UK and Scotland.

 She has appeared on the Today Show and on ITV in Europe.
</item 10>
=== END ITEM id=f40ead54ed417d5b ===

=== ITEM id=3af3dc1d51997bee ===
Gold sentences and facts:
Sentence 0: The 1892 Auburn Tigers football team represented Auburn University in the 1892 college football season.
  Fact 0.0: The 1892 Auburn Tigers football team represented Auburn University.
  Fact 0.1: The 1892 Auburn Tigers football team played in the 1892 college football season.
Sentence 1:  It was the first college football team fielded by the Agricultural and Mechanical College of Alabama, now known as Auburn University.
  Fact 1.0: The 1892 Auburn Tigers football team was the first college football team fielded by the Agricultural and Mechanical College of Alabama.
  Fact 1.1: The Agricultural and Mechanical College of Alabama is now known as Auburn University.
Sentence 2:  With more than 22,000 undergraduate students and a total enrollment of more than 28,000 with 1,260 faculty members, Auburn is the second largest university in Alabama.
  Fact 2.0: Auburn University has more than 22,000 undergraduate students.
  Fact 2.1: Auburn University has a total enrollment of more than 28,000.
  Fact 2.2: Auburn University has 1,260 faculty members.
  Fact 2.3: Auburn University is the second largest university in Alabama.

Retrieved items:
<item 1>
1892 Auburn Tigers football team
The 1892 Auburn Tigers football team represented Auburn University in the 1892 college football season.

 It was the first college football team fielded by the Agricultural and Mechanical College of Alabama, now known as Auburn University.

 The squad was first coached by George Petrie.

 Auburn shut out Georgia Tech, 26–0, just two days after being shut out themselves by North Carolina, 64–0.

 The team finished the season with a record of 2–2.
</item 1>

<item 2>
1997 Alabama Crimson Tide football team
The 1997 Alabama Crimson Tide football team represented the University of Alabama in the college football season of 1997–98.

 The team's head coach was Mike DuBose, who was entering his first year at Alabama.

 They played their home games at both Bryant–Denny Stadium in Tuscaloosa and Legion Field in Birmingham, Alabama and competed in the West Division of the
Southeastern Conference.

 Alabama would finish with a record of 4–7 record in suffering the program's first losing season since the 1984 season.

 The loss against Kentucky marked Alabama's first ever overtime game, as overtime rules for college football had gone into effect the previous season.
</item 2>

<item 3>
1983 Alabama Crimson Tide football team
The 1983 Alabama Crimson Tide football team (variously "Alabama", "UA", "Bama" or "The Tide") represented the University of Alabama in the 1983 NCAA Division
I-A football season.

 It was the Crimson Tide's 91st overall and 50th season as a member of the Southeastern Conference (SEC).

 The team was led by head coach Ray Perkins, in his first year, and played their home games at both Bryant–Denny Stadium in Tuscaloosa and Legion Field in
Birmingham, Alabama.

 They finished the season with a record of eight wins and four losses (8–4 overall, 4–2 in the SEC).

 Ray Perkins, who played as a wide receiver for Bear Bryant in the 1960s, was named as the new head coach at Alabama on December 14, 1982, to succeed Bryant
after his 25-year tenure as Alabama's head coach.
</item 3>

<item 4>
2012 Texas A&amp;M vs. Alabama football game
The 2012 Texas A&M vs. Alabama football game was a college football game between the Texas A&M Aggies and Alabama Crimson Tide at Bryant–Denny Stadium in
Tuscaloosa, Alabama.

 This was the game where Texas A&M freshman quarterback and eventual Heisman Trophy winner Johnny Manziel completed 24 of 31 passes, had 253 passing yards, and
two passing touchdowns with 92 rushing yards to help No. 15 Texas A&M upset No. 1 Alabama 29–24, which led him to being the first freshman to win the Heisman
Trophy.
</item 4>

<item 5>
1905 Alabama Crimson White football team
The 1905 Alabama Crimson White football team (variously "Alabama", "UA" or "Bama") represented the University of Alabama in the 1905 Southern Intercollegiate
Athletic Association football season.

 It was Alabama's 13th overall and 10th season as a member of the Southern Intercollegiate Athletic Association (SIAA).

 The team was led by head coach Jack Leavenworth, in his first year, and played their home games at both the University of Alabama Quad in Tuscaloosa and the
Birmingham Fairgrounds in Birmingham, Alabama.

 They finished the season with a record of six wins and four losses (6–4 overall, 4–4 in the SIAA).
</item 5>

<item 6>
The entries span college and professional football, plus other athletics. Georgia Southern won six FCS titles, 10 SoCon championships, and the Sun Belt outright
in its 2014 FBS debut. Featured season records include Alabama (1997: 4–7; 1983: 8–4; 1906: 5–1; 1905: 6–4), Auburn’s inaugural team (1892: 2–2), and
Connecticut Aggies (189
</item 6>

<item 7>
1906 Alabama Crimson White football team
The 1906 Alabama Crimson White football team (variously "Alabama", "UA" or "Bama") represented the University of Alabama in the 1906 Southern Intercollegiate
Athletic Association football season.

 It was Alabama's 14th overall and 11th season as a member of the Southern Intercollegiate Athletic Association (SIAA).

 The team was led by head coach J. W. H. Pollard, in his first year, and played their home games at both the University of Alabama Quad in Tuscaloosa and the
Birmingham Fairgrounds in Birmingham, Alabama.

 They finished the season with a record of five wins and one loss (5–1 overall, 3–1 in the SIAA).
</item 7>

<item 8>
1902 Connecticut Aggies football team
The 1902 Connecticut Aggies football team represented Connecticut Agricultural College, now the University of Connecticut, in the 1902 college football season.

 This was the seventh year that the school fielded a football team.

 The Aggies were led by first year head coach Edwin O. Smith, and completed the season with a record of 4–3.
</item 8>

<item 9>
Auburn University
Auburn University (AU or Auburn) is a public research university in Auburn, Alabama, United States.

 With more than 22,000 undergraduate students and a total enrollment of more than 28,000 with 1,260 faculty members, Auburn is the second largest university in
Alabama.

 Auburn University is one of the state's two public flagship universities.
</item 9>

<item 10>
West Alabama Tigers football
The West Alabama Tigers football program is the intercollegiate American football team for the University of West Alabama located in the U.S. state of Alabama.

 The team competes in the NCAA Division II and are members of the Gulf South Conference.

 West Alabama's first football team was fielded in 1938.

 The team plays its home games at the 7,000 seat Tiger Stadium in Livingston, Alabama.

 The Tigers are coached by Brett Gilliland.
</item 10>
=== END ITEM id=3af3dc1d51997bee ===

=== ITEM id=80280ce0d61061f7 ===
Gold sentences and facts:
Sentence 0:  Built in 1988 by the TOGO company, it was the fastest roller coaster in the world when it was built (taking the record from American Eagle at Six Flags Great America).
  Fact 0.0: The roller coaster was built in 1988 by the TOGO company.
  Fact 0.1: The roller coaster built by the TOGO company was the fastest roller coaster in the world when it was built.
  Fact 0.2: The roller coaster built by the TOGO company took the record for fastest roller coaster in the world from American Eagle at Six Flags Great America.
Sentence 1: TOGO (株式会社トーゴ , Kabushiki-gaisha Tōgo ) was a Japanese amusement ride company that built roller coasters, giant wheels, carousels, flumes, dark rides, sky cycles and other amusement rides.
  Fact 1.0: TOGO (株式会社トーゴ, Kabushiki-gaisha Tōgo) was a Japanese amusement ride company.
  Fact 1.1: TOGO built roller coasters.
  Fact 1.2: TOGO built giant wheels.
  Fact 1.3: TOGO built carousels.
  Fact 1.4: TOGO built flumes.
  Fact 1.5: TOGO built dark rides.
  Fact 1.6: TOGO built sky cycles.
  Fact 1.7: TOGO built other amusement rides.

Retrieved items:
<item 1>
Anton Schwarzkopf
Anton Schwarzkopf (8 July 1924 – 30 July 2001) was a German engineer of amusement rides, and founder of the Schwarzkopf Industries Company, which built numerous
amusement rides and large roller coasters for both amusement parks and traveling funfairs.
</item 1>

<item 2>
Graph query row:
entity: Bandit
relationship: CREATED_BY
related: TOGO
</item 2>

<item 3>
Millennium Force
Millennium Force is a steel roller coaster built by Intamin at Cedar Point amusement park in Sandusky, Ohio, United States.

 It was the fourteenth roller coaster to be built at the park since Blue Streak opened in 1964.

 Upon completion in 2000, Millennium Force broke six world records and was the world's first Giga Coaster, a roller coaster that exceeds 300 ft in height and
completes a full circuit.

 It was briefly the tallest and fastest in the world until Steel Dragon 2000 opened later the same year.

 The ride is also the third-longest roller coaster in North America after The Beast at Kings Island and Fury 325 at Carowinds.

 It was the first roller coaster to use a cable lift system rather than a traditional chain lift.

 The coaster has a 310 ft , 45-degree lift hill with a 300 ft drop and features two tunnels, three overbanked turns, and four hills.

 It has a top speed of 93 mph .
</item 3>

<item 4>
Giovanola
Giovanola Freres SA was one of the prominent thrill ride manufacturers in the world.

 It was well known for thrill rides and also built electrical power stations, water storage tanks, pipelines, highway bridges, and many other steel products.

 The company started out as a small metal forging shop, founded by Joseph Giovanola in 1888.

 It served as a subcontractor to Intamin supplying rides and roller coasters.

 In 1998 Giovanola started marketing directly under the name Giovanola Amusement Rides Worldwide.

 The company was based in Monthey, Switzerland.

 During its last years in business, Giovanola fashioned steel behind the scenes for companies such as Intamin and Bolliger & Mabillard.

 The company also built its own thrill rides and roller coasters from 1998 to 2001.

 Examples of Giovanola roller coasters are Goliath at Six Flags Magic Mountain, Titan at Six Flags Over Texas and Anaconda at South Africa's Gold Reef City.
</item 4>

<item 5>
Pretzel Amusement Ride Company
The Pretzel Amusement Ride Company was a famous manufacturer of pretzel dark rides. Pretzel built over 1400 pretzel rides and sold them to carnivals and parks.

 The pretzel ride was invented by Marvin Rempfer.

 Leon Cassidy was Marvin's partner in the Pretzel Amusement Ride Company.

 Both names are on the patent.

 Leon Cassidy patented the single-rail dark ride in 1928 along with Marvin Rempfer.

 The company originated in Tumbling Dam Park on the banks of Sunset Lake in Bridgeton, New Jersey.

 A rider said that ""It felt like I was turned and twisted like a Pretzel"", so the name Pretzel was chosen.

 A large heavy pretzel design was originally affixed to the front of each car to prevent the car from flipping backwards.

 In 1929, a standard Pretzel ride had five cars, 350 feet of track, and was one and a half minutes per ride.

 A pretzel ride sold for $1,200.
</item 5>

<item 6>
White Cyclone
White Cyclone (ホワイトサイクロン , Howaito Saikuron ) is a wooden roller coaster at Nagashima Spa Land in Mie Prefecture, Japan.

 At 1700 m in length, White Cyclone is the third longest wooden roller coaster in the world, and is the longest wooden roller coaster outside of the United
States.

 Despite its length, White Cyclone is still considerably shorter than the 2479 m Steel Dragon 2000, the world's longest steel roller coaster, which is also at
Nagashima Spa Land.

 In addition to being the third longest wooden roller coaster, White Cyclone is the seventh tallest wooden roller coaster in the world and the fourth tallest
wooden roller coaster outside the United States.

 A single ride on the White Cyclone costs ¥1,000 (approximately $9 USD), and the ride is restricted to those individuals above 1.3 m in height; and those
individuals under 54 years of age.
</item 6>

<item 7>
Castle Park (amusement park)
Castle Park, formerly Castle Amusement Park, is a 25-acre amusement park and family amusement center located in Riverside, California.

 The park utilizes a medieval "castle" theme and includes attractions such as a miniature golf course, arcade, and 27 amusement rides including three roller
coasters such as "Merlin's Revenge", a junior rollercoaster, "Screamin' Demon" a spinning Wild Mouse rollercoaster, and "Little Dipper", a children's
rollercoaster.

 The main "castle" themed building, houses the arcade as well as its only dark ride; "Ghost Blasters", an interactive attraction, designed by Sally Corporation,
which can also be found at other amusement parks throughout North America.

 The park was designed, built and operated by Bud Hurlbut, who designed several rides at Knott's Berry Farm.

 Castle Park is currently owned and operated by Palace Entertainment.
</item 7>

<item 8>
Bandit (Yomiuriland)
Bandit (バンデット , Bandetto ) is a steel roller coaster located at Yomiuriland in the city of Inagi, near Tokyo, Japan.

 Built in 1988 by the TOGO company, it was the fastest roller coaster in the world when it was built (taking the record from American Eagle at Six Flags Great
America).

 It lost the record to Magnum XL-200 at Cedar Point one year later.
</item 8>

<item 9>
Mack Rides
Mack Rides GmbH & Co KG, also known simply as Mack Rides, is a German company that designs and constructs amusement rides.

 Mack Rides in headquartered in Waldkirch, Baden-Württemberg, Germany.

 Mack Rides is one of the world's oldest amusement industry suppliers and builds all kinds of amusement devices including several types of flat rides, dark
rides, log flumes, tow boat rides and roller coasters.

 The family that owns Mack Rides also owns Europa-Park.
</item 9>

<item 10>
TOGO
TOGO (株式会社トーゴ , Kabushiki-gaisha Tōgo ) was a Japanese amusement ride company that built roller coasters, giant wheels, carousels, flumes, dark rides, sky
cycles and other amusement rides.
</item 10>
=== END ITEM id=80280ce0d61061f7 ===
