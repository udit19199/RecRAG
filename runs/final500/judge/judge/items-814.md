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

This file has 6 items.

=== ITEM id=673f4a4e69677268 ===
Gold sentences and facts:
Sentence 0: The video takes place with the band and the album 's two protagonists Christian ( Josh Boswell ) and Gloria ( Lisa Stelly ) taking refuge in a white room after robbing a bank . The police arrive outside the room and open fire through the window , to the couple 's terror . As bullets rain through the room , the band continues playing . Gloria picks up the phone and throws it into a fish tank . As the bullets continue to fly and tear apart the room , Christian and Gloria become calm and walk toward each other , unharmed by the bullets . They embrace and kiss as the room goes dark , recreating the 21st Century Breakdown cover art . As the song ends , the room lights up again , and they are still uninjured . After the room lights up , there are various shots of destruction in the room , including a shot of some of the writing on the walls . The writing includes an excerpt of the lyrics to `` 21 Guns '' , as well as those of the song `` See The Light '' which also appears on `` 21st Century Breakdown '' . The video may represent the meaning of the song in the 21st Century Breakdown story .
  Fact 0.0: The video features the band
  Fact 0.1: The video features the album's two protagonists, Christian and Gloria
  Fact 0.2: Christian is played by Josh Boswell
  Fact 0.3: Gloria is played by Lisa Stelly
  Fact 0.4: Christian and Gloria take refuge in a white room after robbing a bank
  Fact 0.5: The police arrive outside the room and open fire through the window
  Fact 0.6: The police fire to the couple's terror
  Fact 0.7: As bullets rain through the room, the band continues playing
  Fact 0.8: Gloria picks up the phone and throws it into a fish tank
  Fact 0.9: As the bullets continue to fly and tear apart the room, Christian and Gloria become calm
  Fact 0.10: Christian and Gloria walk toward each other, unharmed by the bullets
  Fact 0.11: Christian and Gloria embrace and kiss as the room goes dark
  Fact 0.12: The embrace and kiss recreate the 21st Century Breakdown cover art
  Fact 0.13: As the song ends, the room lights up again
  Fact 0.14: Christian and Gloria are still uninjured when the room lights up
  Fact 0.15: After the room lights up, there are various shots of destruction in the room
  Fact 0.16: The shots include a shot of some of the writing on the walls
  Fact 0.17: The writing includes an excerpt of the lyrics to 21 Guns
  Fact 0.18: The writing also includes lyrics of the song See The Light
  Fact 0.19: The song See The Light also appears on 21st Century Breakdown
  Fact 0.20: The video may represent the meaning of the song in the 21st Century Breakdown story

Retrieved items:
<item 1>
21 Guns (song)
British Phonographic Industry . Enter 21 Guns in the search field and then press Enter . Jump up ^ `` American single certifications -- Green Day -- 21 Guns ''
. Recording Industry Association of America . Retrieved October 13 , 2015 . If necessary , click Advanced , then click Format , then select Single , then click
SEARCH External links Lyrics of this song at MetroLyrics ( hide ) Green Day Billie Joe Armstrong Mike Dirnt Tré Cool Jason White John Kiffmeyer Studio albums 39
/ Smooth Kerplunk Dookie Insomniac Nimrod Warning American Idiot 21st Century Breakdown ¡ Uno ! ¡ Dos ! ¡ Tré ! Revolution Radio Live albums Live Tracks Tune In
, Tokyo ... Bullet in a Bible Awesome as Fuck Compilation albums 1,039 / Smoothed Out Slappy Hours International Superhits ! Shenanigans Green Day Demolicious
Greatest Hits : God 's Favorite Band EPs 1,000 Hours Slappy Sweet Children Last Night on Earth : Live in Tokyo Singles `` Longview '' `` Welcome to Paradise ''
`` Basket Case '' `` When I Come Around '' `` She '' `` J.A.R. '' `` Geek
</item 1>

<item 2>
21 Guns (song)
21 Guns ( song ) - wikipedia 21 Guns ( song ) Jump to : navigation , search `` 21 Guns '' Single by Green Day from the album 21st Century Breakdown Released May
25 , 2009 ( digital download ) July 14 , 2009 ( CD single ) Format Digital download , CD single , vinyl Recorded 2008 Genre Alternative rock Length 5 : 21 Label
Reprise Songwriter ( s ) Billie Joe Armstrong , David Bowie , Mike Dirnt , and Tré Cool Producer ( s ) Butch Vig and Green Day Green Day singles chronology ``
Know Your Enemy '' ( 2009 ) `` 21 Guns '' ( 2009 ) `` East Jesus Nowhere '' ( 2009 ) `` Know Your Enemy '' ( 2009 ) `` 21 Guns '' ( 2009 ) `` East Jesus Nowhere
'' ( 2009 ) Audio sample file help `` 21 Guns '' is a song by American punk rock band Green Day . It was released as the second single from their eighth album ,
21st Century Breakdown .
</item 2>

<item 3>
21 Guns (song)
The video was released February 1 . The second version of the song with the cast of American Idiot features Stark Sands , John Gallagher Jr. , and Michael Esper
singing their respective solos instead of Armstrong . It was performed with Armstrong singing lead male vocals at the 52nd Grammy Awards . Live EP 21 Guns Live
EP EP by Green Day Released September 4 , 2009 Recorded July 25 , 2009 in Albany , New York ; July 28 , 2009 Madison Square Garden , New York Genre Pop punk ,
punk rock , alternative rock Length 17 : 52 Label Reprise Producer Green Day Green Day released a live EP for the song including live performances of `` Welcome
to Paradise '' , `` Brain Stew / Jaded '' and `` F.O.D. '' recorded live in Albany , New York and Madison Square Garden . This EP was released on the Australian
iTunes store , Amazon UK , and Napster . Track listing Original version : Digital download / Maxi CD No . Title Length 1 . `` 21 Guns '' 5 : 21 2 .
</item 3>

<item 4>
21 Guns (song)
Retrieved June 16 , 2012 . Jump up ^ `` Green Day -- 21 Guns ( CD ) at Discogs '' . Discogs . Retrieved June 16 , 2012 . Jump up ^ `` Transformers : Revenge of
the Fallen Soundtrack '' . digiRAMA . Archived from the original on June 29 , 2009 . Retrieved July 12 , 2010 . Jump up ^ `` MTV Games , Harmonix & Warner Bros.
/ Reprise Records Partner With Green Day to Bring Their Music Exclusively to Rock Band / / News / / Rock Band '' . Rockband.com . Retrieved August 21 , 2009 .
Jump up ^ `` SkinnyLove21 's Journal -- The Vampire Diaries Soundtrack , Season 1 '' . last.fm. April 16 , 2010 . Archived from the original on August 18 , 2010
. Retrieved July 12 , 2010 . Jump up ^ `` '' 21 Guns '' -- Green Day `` . rockband.com . Archived from the original on April 25 , 2010 .
</item 4>

<item 5>
21 Guns (song)
Title Length 1 . `` 21 Guns '' 5 : 21 2 . `` Favorite Son '' ( Green Day 's contribution to the Rock Against Bush Vol. 2 compilation ) 2 : 13 3 . `` 21 Guns ''
( Studio 880 version ) 5 : 17 CD single / 7 '' single No . Title Length 1 . `` 21 Guns '' 5 : 21 2 . `` Favorite Son '' ( From `` Rock Against Bush Vol. 2 '' )
2 : 13 Promo CD No . Title Length 1 . `` 21 Guns '' ( Radio Edit ) 4 : 37 Musical cast version : Digital download / CD single No . Title Length 1 . `` 21 Guns
'' ( Green Day with the cast of American Idiot version ) 4 : 41 21 Guns EP No . Title Length 1 . `` 21 Guns '' ( Green Day with the cast of American Idiot
version ) 4 : 41 2 . `` 21 Guns '' ( Original version ) 5 : 21 3 .
</item 5>

<item 6>
21 Guns (song)
The video may represent the meaning of the song in the 21st Century Breakdown story . Musical cast version `` 21 Guns '' Single by Green Day and the cast of
American Idiot from the album American Idiot : The Original Broadway Cast Recording B - side `` Like a Rolling Stone ( Bob Dylan Cover ) '' Released December 3
, 2009 Format Digital download , CD single Recorded 2009 Genre Alternative rock Length 4 : 41 Label Reprise Songwriter ( s ) Billie Joe Armstrong , David Bowie
, Mike Dirnt , and Tré Cool Producer ( s ) Green Day Green Day and the cast of American Idiot singles chronology `` 21 Guns '' ( 2009 ) `` When It 's Time '' (
2010 ) `` 21 Guns ( Musical Cast Version ) '' ( 2010 ) `` When It 's Time '' ( 2010 ) Green Day has recorded two versions of `` 21 Guns '' with the cast of
American Idiot . The recording was produced by Billie Joe Armstrong . The recording with Armstrong singing the male solo parts was released on December 3 to
Spinner.com and also released for digital download on December 22 .
</item 6>

<item 7>
21 Guns (song)
It became available for download on the music video game series Rock Band on July 7 , 2009 , along with the songs `` East Jesus Nowhere '' and `` Know Your
Enemy . The song was featured in episode 6 of The Vampire Diaries . The song is featured in the video game Green Day : Rock Band and is downloadable content for
the Rock Band series . A live version was also released on 21 Guns Live EP in September 2009 . Music video A music video was directed by Marc Webb and filmed in
Los Angeles on June 6 , 2009 . It premiered on MySpace on Monday , June 22 , 2009 at midnight EDT . A shorter version of the video also exists , which is set to
the radio edit of the song . It is the third Green Day video to feature touring guitarist Jason White performing with the band , following `` Wake Me Up When
September Ends '' and `` Working Class Hero '' . He is also seen briefly in the music videos for `` When I Come Around '' and `` Jesus of Suburbia '' but he is
not performing with the band .
</item 7>

<item 8>
21 Guns (song)
GreenDay.com . November 13 , 2009 . Archived from the original on April 13 , 2010 . Retrieved July 12 , 2010 . Jump up ^ `` Press Release '' . GreenDay.com .
December 2 , 2009 . Archived from the original on June 8 , 2011 . Retrieved July 12 , 2010 . Jump up ^ Officially released video Green Day -- 21 Guns ( Cast
Version ) on YouTube Jump up ^ `` Get The New 21 Guns EP In Itunes Now '' . GreenDay.com . February 2 , 2010 . Archived from the original on June 8 , 2011 .
Retrieved July 12 , 2010 . Jump up ^ `` Australian-charts.com -- Green Day -- 21 Guns '' . ARIA Top 50 Singles . Retrieved October 27 , 2017 . Jump up ^ ``
Austriancharts.at -- Green Day -- 21 Guns '' ( in German ) . Ö3 Austria Top 40 . Retrieved October 27 , 2017 .
</item 8>

<item 9>
21 Guns (song)
rockband.com . Archived from the original on April 25 , 2010 . Retrieved July 12 , 2010 . Jump up ^ `` 21 Guns Live EP '' . GreenDay.com . September 20 , 2009 .
Archived from the original on April 13 , 2010 . Retrieved July 12 , 2010 . Jump up ^ `` 21 Guns Video '' . Green Day.com . June 5 , 2009 . Archived from the
original on June 8 , 2011 . Retrieved July 12 , 2010 . Jump up ^ `` 2009 MTV Video Music Awards Winners '' . MTV . Retrieved July 12 , 2010 . Jump up ^ `` Green
Day -- making of 21 Guns video '' . MySpace Video . June 23 , 2009 . Archived from the original on July 2 , 2009 . Retrieved July 12 , 2010 . Jump up ^ `` Press
Release '' . GreenDay.com . November 13 , 2009 .
</item 9>

<item 10>
21 Guns (song)
Frustrators The Lookouts The Big Cats The Influents Related articles Discography List of songs Awards and nominations Adeline Records Cigarettes and Valentines
Money Money 2020 Stop Drop and Roll ! ! ! American Edit Green Day 's American Idiot soundtrack Green Day : Rock Band songs ¡ Cuatro ! Foreverly Book Portal
Retrieved from `` https://en.wikipedia.org/w/index.php?title=21_Guns_(song)&oldid=820470590 '' Categories : 2000s ballads 2009 singles Green Day songs Rock
ballads Songs written by Billie Joe Armstrong Song recordings produced by Butch Vig Music videos directed by Marc Webb Songs written by Mike Dirnt 2008 songs
2009 songs Number - one singles in Poland Reprise Records singles Hidden categories : CS1 Italian - language sources ( it ) Wikipedia pages semi-protected from
banned users Use mdy dates from March 2012 Articles with hAudio microformats All articles with unsourced statements Articles with unsourced statements from
October 2010 Music infoboxes with deprecated parameters Singlechart usages for Australia Singlechart usages for Austria Singlechart usages for Flanders
Singlechart usages for
</item 10>
=== END ITEM id=673f4a4e69677268 ===

=== ITEM id=1c65d62ee3369630 ===
Gold sentences and facts:
Sentence 0: In negotiation theory , the best alternative to a negotiated agreement or BATNA is the most advantageous alternative course of action a party can take if negotiations fail and an agreement can not be reached . BATNA is the key focus and the driving force behind a successful negotiator . A party should generally not accept a worse resolution than its BATNA . Care should be taken , however , to ensure that deals are accurately valued , taking into account all considerations , such as relationship value , time value of money and the likelihood that the other party will live up to their side of the bargain . These other considerations are often difficult to value , since they are frequently based on uncertain or qualitative considerations , rather than easily measurable and quantifiable factors .
  Fact 0.0: In negotiation theory, the best alternative to a negotiated agreement (BATNA) is the most advantageous course of action a party can take if negotiations fail and no agreement can be reached.
  Fact 0.1: BATNA is the key focus of a successful negotiator.
  Fact 0.2: BATNA is the driving force behind a successful negotiator.
  Fact 0.3: A party should generally not accept a resolution worse than its BATNA.
  Fact 0.4: Care should be taken to ensure that deals are accurately valued.
  Fact 0.5: Deal valuation should take into account all considerations.
  Fact 0.6: Relationship value is a consideration in valuing a deal.
  Fact 0.7: Time value of money is a consideration in valuing a deal.
  Fact 0.8: The likelihood that the other party will live up to its side of the bargain is a consideration in valuing a deal.
  Fact 0.9: The other considerations are often difficult to value.
  Fact 0.10: The other considerations are frequently based on uncertain or qualitative considerations.
  Fact 0.11: The other considerations are qualitative rather than easily measurable and quantifiable factors.

Retrieved items:
<item 1>
Best alternative to a negotiated agreement
History ( edit ) BATNA was developed by negotiation researchers Roger Fisher and William Ury of the Harvard Program on Negotiation ( PON ) , in their series of
books on Principled negotiation that started with Getting to YES , unwittingly duplicating the game theory concept of a disagreement point from bargaining
problems pioneered by Nobel Laureate John Forbes Nash decades earlier . A Nash Equilibrium is reached among a group of players when no player can benefit from
changing strategies if every other player sticks to their current strategy . For example , Amy and Phil are in Nash Equilibrium if Amy is making the best
decision she can , taking into account Phil 's decision , and Phil is making the best decision he can , taking into account Amy 's decision . Likewise , a group
of players are in Nash Equilibrium if each one is making the best decision that he or she can , taking into account the decisions of the others .
</item 1>

<item 2>
Best alternative to a negotiated agreement
See also ( edit ) Conflict resolution research Opportunity cost References ( edit ) Jump up ^ Best Alternative to a Negotiated Agreement Negotiation Experts
Jump up ^ Best Alternative to a Negotiated Agreement YArooms Jump up ^ Negotiation , Readings , Exercises and Cases , Roy J. Lewicki ^ Jump up to : Nolan -
Haley , Jaqueline M ( 2001 ) . Alternative Dispute Resolution in a Nutshell . Thomson West . pp. 39 -- 50 . ISBN 978 - 0 - 314 - 18014 - 8 . ^ Jump up to :
Fisher and Ury , Roger and William ( 2011 ) . Getting to YES . Penguin Books . pp. 1 -- 170 . ISBN 978 - 0 - 14 - 311875 - 6 . Jump up ^ Gulliver , P.H ( 1979 )
. Disputes and Negotiation : A Cross Culture Perspective . Academic Press . p. 287 . Jump up ^ Morris , Michael W. ; Gelfand , Michele J. ( 2004 ) .
</item 2>

<item 3>
Best alternative to a negotiated agreement
, a non-profit organization . About Wikipedia
</item 3>

<item 4>
Best alternative to a negotiated agreement
Retrieved 1 October 2012 . External links ( edit ) BATNA Explained by David Venter Alternative definition of BATNA Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Best_alternative_to_a_negotiated_agreement&oldid=819966880 '' Categories : Negotiation Dispute resolution Business
terms Hidden categories : Articles needing more detailed references Articles needing additional references from July 2013 All articles needing additional
references All articles lacking reliable references Articles lacking reliable references from July 2013 All articles with unsourced statements Articles with
unsourced statements from October 2007 Articles lacking reliable references from February 2015 Wikipedia articles needing page number citations from July 2013
Talk Contents About Wikipedia Deutsch Español Հայերեն Polski Edit links This page was last edited on 12 January 2018 , at 08 : 19 . Text is available under the
Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy .
Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization .
</item 4>

<item 5>
Best alternative to a negotiated agreement
Contents ( hide ) 1 Overview 2 History 3 See also 4 References 5 External links Overview ( edit ) The BATNA is often seen by negotiators not as a safety net ,
but rather as a point of leverage in negotiations . Although a negotiator 's alternative options should , in theory , be straightforward to evaluate , the
effort to understand which alternative represents a party 's BATNA is often not invested . Options need to be actual and actionable to be of value , however
without the investment of time , options will frequently be included that fail on one of these criteria . Most managers overestimate their BATNA whilst
simultaneously investing too little time into researching their real options . This can result in poor or faulty decision making and negotiating result .
Negotiators also need to be aware of the other negotiator 's BATNA and to identify how it compares to what they are offering . Some people may adopt aggressive
, coercive , threatening and / or deceptive techniques , this is known as a hard negotiation style ; a theoretical example of this is adversarial approach style
negotiation . Others may employ a soft style , which is friendly , trusting , compromising , and conflict avoiding .
</item 5>

<item 6>
Best alternative to a negotiated agreement
Others may employ a soft style , which is friendly , trusting , compromising , and conflict avoiding . According to Fisher and Ury , when hard negotiators meet
soft negotiators , the hard negotiators usually win their position , but at the cost of potentially damaging the long term relationship between the parties .
Attractive alternatives are needed to develop a strong BATNA . In the best - selling book Getting to YES : Negotiating Agreement Without Giving In , the authors
give 3 suggestions of how to accomplish this : Inventing a list of actions one might take if no agreement is reached Converting some of the more promising ideas
and transforming them into tangible and partial alternatives Selecting the alternative that sounds best In negotiations involving different cultures , all
parties need to account for cultural cognitive behaviors and should not let judgments and biases affect the negotiation . The individual should be separate from
the objective . The purpose here , as Gulliver mentions , is for negotiation parties to be aware . Preparation at all levels , including prejudice - free
thoughts , emotion - free behavior , bias - free behavior are helpful according to Morris and Gelfand .
</item 6>

<item 7>
Bond (finance)
An alternative process for bond issuance , which is commonly used for smaller issues and avoids this cost , is the private placement bond . Bonds sold directly
to buyers may not be tradeable in the bond market . Historically an alternative practice of issuance was for the borrowing government authority to issue bonds
over a period of time , usually at a fixed price , with volumes sold on a particular day dependent on market conditions . This was called a tap issue or bond
tap . Features ( edit ) Principal ( edit ) 1978 $1,000 U.S. Treasury Bond Nominal , principal , par , or face amount is the amount on which the issuer pays
interest , and which , most commonly , has to be repaid at the end of the term . Some structured bonds can have a redemption amount which is different from the
face amount and can be linked to performance of particular assets . Maturity ( edit ) The issuer has to repay the nominal amount on the maturity date . As long
as all due payments have been made , the issuer has no further obligations to the bond holders after the maturity date .
</item 7>

<item 8>
Batman: Knightfall
It also includes a selection of challenge maps featuring characters and enemies from the storyline . Bane makes reference to `` breaking the bat '' on a few
occasions and in multiplayer , Bane can perform the backbreaker to instantly kill Batman or Robin . In Batman : Arkham Knight , Azrael appears in a sidequest in
which he expresses his interest in becoming Batman 's successor ( though this is the Michael Lane version ) , referencing Azrael replacing Batman in the
Knightfall storyline . In the game 's ending , Batman activates the Knightfall protocol after Scarecrow reveals to the world he 's Bruce Wayne . He is
supposedly killed when he arrives home and Wayne Manor explodes ( though a likely scenario is that he faked his death ) and a new Batman emerged protecting
Gotham 's streets afterwards . Whether this is Bruce or a successor is never shown .
</item 8>

<item 9>
Best alternative to a negotiated agreement
( 2004 ) . `` Cultural Differences and Cognitive Dynamics : Expanding the Cognitive Perspective on Negotiation '' . In Gelfand , Michele J. ; Brett , Jeanne M.
The Handbook of Negotiation and Culture . Stanford University Press . pp. 45 -- 70 . ISBN 978 - 0 - 8047 - 4586 - 4 . Jump up ^ Myerson , Roger B. `` Nash
Equilibrium and the History of Economic Theory '' ( PDF ) . Nash Equilibrium and the History of Economic Theory . Journal of Economic Literature . Retrieved 1
October 2012 . Jump up ^ Hawkins and Steiner , Jeff and Neil . `` The Nash Equilibrium Meets Batna '' ( PDF ) . Gamed Therory Varied Uses in ADR . Harvard
University Press . Archived from the original ( PDF ) on February 28 , 2013 . Retrieved 1 October 2012 .
</item 9>

<item 10>
Best alternative to a negotiated agreement
Best alternative to a negotiated agreement - wikipedia Best alternative to a negotiated agreement Jump to : navigation , search This article needs additional
citations for verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . (
July 2013 ) ( Learn how and when to remove this template message ) In negotiation theory , the best alternative to a negotiated agreement or BATNA is the most
advantageous alternative course of action a party can take if negotiations fail and an agreement can not be reached . BATNA is the key focus and the driving
force behind a successful negotiator . A party should generally not accept a worse resolution than its BATNA . Care should be taken , however , to ensure that
deals are accurately valued , taking into account all considerations , such as relationship value , time value of money and the likelihood that the other party
will live up to their side of the bargain . These other considerations are often difficult to value , since they are frequently based on uncertain or
qualitative considerations , rather than easily measurable and quantifiable factors .
</item 10>
=== END ITEM id=1c65d62ee3369630 ===

=== ITEM id=528b98aecf08fe69 ===
Gold sentences and facts:
Sentence 0: No . Cloverway inc . title Original Japanese and Viz Media titles Episode director ( s ) Writer ( s ) Art director ( s ) Animation director ( s ) Original airdate English airdate ( United States ) 90 `` Star Struck , Bad Luck '' `` Premonition of the Apocalypse : The Mysterious New Guardians Appear '' `` Chikyū hōkai no yokan ? Nazo no shin Senshi shutsugen '' ( 地球 崩壊 の 予感 ? 謎 の 新 戦士 出現 ) Junichi Sato Sukehiro Tomita Kenichi Tajiri Masahiro Andō March 19 , 1994 June 12 , 2000 After experiencing apocalyptic visions , Rei is attacked by a powerful monster called a Daimon that Sailor Moon is unable to fight after it disables her brooch , reversing her transformation . The Daimon is defeated by a pair of mysterious new warriors , leaving the girls to despondently realize that a new conflict has begun . 91 `` Crystal Clear Again '' `` The Rod of Love Is Born : Usagi 's New Transformation '' `` Ai no roddo tanjō ! Usagi no shin henshin '' ( 愛 の ロッド 誕生 ! うさぎ の 新 変身 ) Yūji Endō Shigeru Yanagawa Yoshiyuki Shikano Katsumi Tamegai March 26 , 1994 June 13 , 2000 While Mamoru strives to cheer Usagi up after her failure in the previous battle , a young girl is the latest target of the Death Busters . Due to Usagi 's inability to transform , the under - powered Sailor Team struggles to fight off the Daimon and are once again forced to rely on the mysterious two warriors . Once they determine the girl 's pure heart is not a talisman , they leave the Sailor Team to fend for themselves . Meanwhile , Tuxedo Mask and Usagi hold hands after the latter 's refusal to escape , and through the power of their love , Usagi 's disabled Crystal Star brooch transforms into the Cosmic Heart Compact , and she gains the Spiral Heart Moon Rod and , along with it , the ability to transform and defeat the new enemy . 92 `` Driving Dangerously '' `` A Handsome Boy ? Haruka Tenoh 's Secret '' `` Suteki na bishōnen ? Ten'ō Haruka no himitsu '' ( 素敵 な 美少年 ? 天王 はるか の 秘密 ) Kunihiko Ikuhara Katsuyuki Sumisawa Kazuhisa Asai Hisashi Kagawa April 16 , 1994 June 14 , 2000 Avoiding their study group , Usagi and Minako meet Haruka Tenou , an attractive and supposedly male student of a prestigious local school , with whom they both become infatuated . After Minako is defeated in a racing game , Michiru arrives to pick up Haruka , and the two depart with Usagi and Minako soon following behind them . Once Michiru and Haruka separate at a mechanic 's shop , Usagi and Minako take this as their chance , and Haruka tells them about `` him '' - self until a Daimon arrives to attack Kameda , the shop mechanic . The Daimon starts to escape when Usagi and Minako transform into their Guardian forms to chase it down , when Uranus and Neptune arrive to investigate Kameda 's pure heart crystal . When it turns out not to be a talisman they retreat and Sailor Moon defeats the Daimon . Later , the rest of the girls are at Crown Game Center with Usagi and Minako in hopes of seeing the cute boy they 'd told them about ; Michiru arrives with a blonde girl , whom Usagi and Minako initially fail to realize is Haruka ; once they do , Minako falls to the ground in anguish , and Haruka notes that she 'd never said she was male . 93 `` Bad Harmony '' `` Usagi 's Idol : The Graceful Genius Michiru '' `` Usagi no akogare ! Yūbi no tensai Michiru '' ( うさぎ の 憧れ ! 優美 な 天才 みちる ) Hiroki Shibata Megumi Sugihara Kazuyuki Hashimoto Taichi Nakamura April 23 , 1994 June 15 , 2000 When Usagi is concerned that she is not smart enough to be with Mamoru after being scolded by the other girls , she seeks advice from Michiru Kaioh , who suggests that Usagi and Mamoru go to a violin recital . 94 `` Swept Off Her Feet '' `` Protect the Pure Heart : The Three - Way Battle '' `` Pyua na kokoro wo mamore ! Teki mikata mittsu tomoe ransen '' ( 純 な 心 を 守れ ! 敵 味方 三つ巴 乱戦 ) Noriyo Sasaki Sukehiro Tomita Kenichi Tajiri Ikuko Itō April 30 , 1994 June 16 , 2000 Usagi 's friend , Unazuki , is attacked by a Daimon and has her pure heart crystal stolen . But when Sailor Moon tries to take the crystal back to Unazuki , Uranus and Neptune attempt to steal it in order to claim the Talisman . 95 `` Blinded By Love 's Light '' `` Let Moon Help With Your Love Problems '' `` Koi no otasuke wa Mūn ni omakase '' ( 恋 の おたすけ は ムーン に おまかせ ) Harume Kosaka Shigeru Yanagawa Yoshiyuki Shikano Mari Tominaga May 7 , 1994 June 19 , 2000 Naru and Umino enter an affection contest in order to prove their strong mutual romantic love for each other , but find themselves targeted by a Daimon . 96 `` Lita Borrows Trouble '' `` Coldhearted Uranus : Makoto in Danger '' `` Reikoku na Uranusu ? Makoto no pinchi '' ( 冷酷 な ウラヌス ? まこと の ピンチ ) Kōnosuke Uda Yoji Enokido Kazuhisa Asai Masahiro Andō May 14 , 1994 June 20 , 2000 Makoto is targeted by a Daimon , but manages to temporarily evade it . Keen to analyze her pure heart crystal , Haruka befriends Makoto in order to get close to her . 97 `` Damp Spirits '' `` The Labyrinth of Water : Ami Targeted '' `` Mizu no rabarinsu ! Nerawareta Ami '' ( 水 の ラビリンス ! ねら われ た 亜美 ) Junichi Sato Megumi Sugihara Kazuyuki Hashimoto Katsumi Tamegai May 21 , 1994 June 21 , 2000 Ami 's desire to avoid competition leads her to offending Michiru inadvertently in a swimming race , as well as Usagi accidentally hurting Ami 's feelings . Meanwhile , a Daimon selects Ami as a target . 98 `` Friendly Foes '' `` To Save Our Friends : Moon and Uranus Join Forces '' `` Tomodachi wo sukue ! Mūn Uranusu rengō '' ( 友達 を 救え ! ムーン ウラヌス 連合 ) Yūji Endō Sukehiro Tomita Kenichi Tajiri Taichi Nakamura May 28 , 1994 June 22 , 2000 After Sailor Neptune is captured and possibly killed by Kaolinite , Sailor Moon and Sailor Uranus find themselves shackled together and forced to reassess their less - than - friendly attitudes toward each other in order to save Sailor Neptune from further harm . 99 `` Mixed Emotions '' `` A Man 's Kindness : Yuichiro Heartbroken by Rei '' `` Otoko no yasashisa ! Yūichirō Rei ni shitsuren ? '' ( 男 の 優し さ ! 雄一郎 レイ に 失恋 ? ) Harume Kosaka Megumi Sugihara Yoshiyuki Shikano Shinya Hasegawa June 18 , 1994 June 23 , 2000 Worried by Rei 's intensive meditation , Yuuichirou becomes concerned that she has fallen deeply head - over-heels in love for another man . When he sees her with Haruka and mistakes her for a man , Yuuichirou is convinced that `` he '' is Rei 's new romantic suitor . 100 `` Individual Happiness '' `` I Want to Quit Being a Sailor Guardian : Minako 's Dilemma '' `` Sērā Senshi wo yametai ! ? Minako no nayami '' ( S 戦士 を 辞め たい ! ? 美奈子 の 悩み ) Hiroki Shibata Yoji Enokido Kazuhisa Asai Mari Tominaga June 25 , 1994 June 26 , 2000 Minako is despondent about her double life of being a Sailor Guardian and a normal girl . Meanwhile , an old friend of hers , who she once fell in love with , is attacked by a Daimon , and Minako , as Sailor Venus , must use her skills of volleyball to save him . 101 `` Birthday Blues , Part 1 '' `` Usagi in Tears : a Glass Slipper for My Birthday '' `` Usagi namida ! Tanjōbi ni garasu no kutsu wo '' ( うさぎ 涙 ! 誕生 日 に ガラス の 靴 を ) Yūji Endō Katsuyuki Sumisawa Kazuyuki Hashimoto Masahiro Andō July 2 , 1994 June 27 , 2000 Usagi is furious when Mamoru forgets her 15th birthday . When Mamoru buys her an expensive pair of glass slippers to make up for it , he fails to realize that Kaolinite sabotaged the shoes with a Daimon egg to steal her pure heart crystal . To make matters worse , Kaolinite steals Usagi 's Cosmic Heart Compact leaving her unable to transform , and Tuxedo Mask gets captured . 102 `` Birthday Blues , Part Two '' `` The Stolen Pure Heart : Usagi 's Crisis '' `` Ubawareta pyua na kokoro ! Usagi zettai zetsumei '' ( 奪 われ た 純 な 心 ! うさぎ 絶 体 絶命 ) Noriyo Sasaki Katsuyuki Sumisawa Kenichi Tajiri Taichi Nakamura July 16 , 1994 June 28 , 2000 With the Cosmic Heart Compact stolen and Tuxedo Mask kidnapped , Usagi and the four other Sailor Guardians , along with Sailors Uranus and Neptune , go to retake the compact so Usagi can become Sailor Moon again , rescue Tuxedo Mask and face Kaolinite in a final showdown at Tokyo Tower . 103 `` Hello , Sailor Mini Moon '' `` The Arrival of a Tiny Pretty Guardian '' `` Yatte kita chiccha na Bishōjo Senshi '' ( やって来 た ちっちゃな 美 少女 戦士 ) Kunihiko Ikuhara Shigeru Yanagawa Yoshiyuki Shikano Kazuya Kuroda August 6 , 1994 June 29 , 2000 Eudial , and the Witches 5 are tasked to take over from Kaolinite 's position , in which they target a genius drummer recruited to play at a festival for Rei . The Sailor Guardians can not defeat the Daimon who attacks her , but Chibiusa arrives as the Guardians ' newest ally , Sailor Chibi - Moon . 104 `` Tainted Tea Party '' `` Making New Friends : Chibi Moon 's Adventure '' `` Tomodachi wo motomete ! Chibi Mūn no katsuyaku '' ( 友達 を 求め て ! ちび ムーン の 活躍 ) Junichi Sato Megumi Sugihara Kazuhisa Asai Ikuko Itō August 20 , 1994 June 30 , 2000 Chibiusa , who has returned from Crystal Tokyo for further training , attempts to adapt to her new home by making friends her age . She tries to befriend a master who does tea ceremonies , only to become the Death Busters ' latest target . 105 `` People Who Need People '' `` I Want Power : Mako Lost in Doubt '' `` Pawā ga hoshī ! Mako - chan no mayoi michi '' ( 力 が 欲しい ! まこ ちゃん の 迷い 道 ) Hiroki Shibata Shigeru Yanagawa Yoshiyuki Shikano Mari Tominaga August 27 , 1994 July 3 , 2000 Makoto is training in the mountains , trying to become much stronger after being easily defeated by a Daimon before . Her trainer is attacked , and tells her that nobody can live alone . She attacks the Daimon , fails , and realizes she needs her best friends in order to win . 106 `` Related By Destiny '' `` The Bond of Destiny : Uranus 's Distant Past '' `` Unmei no kizuna ! Uranusu no tōi hi '' ( 運命 の きずな ! ウラヌス の 遠い 日 ) Takuya Igarashi Yoji Enokido Kenichi Tajiri Katsumi Tamegai September 3 , 1994 July 4 , 2000 Haruka remembers the past when she first met Michiru , who told her that she was destined to become Sailor Uranus . Haruka initially refused that destiny , for she loved her life too much and was afraid of change . Haruka was attacked , and Sailor Neptune was injured while defending her . In her anguish , Haruka took the transformation pen and accepted her destiny as Sailor Uranus , the Sailor Guardian of the Sky . 107 `` Art Appreciation '' `` Art Is an Explosion of Love : Chibiusa 's First Love '' `` Geijutsu wa ai no bakuhatsu ! Chibiusa no hatsukoi '' ( 芸術 は 愛 の 爆発 ! ちびうさ の 初恋 ) Yūji Endō Katsuyuki Sumisawa Minoru Ōkōchi Taichi Nakamura September 10 , 1994 July 5 , 2000 Chibiusa meets a boy named Masanori at her art class and develops a crush on him . Unfortunately , he already has a crush on Michiru , who is taking a class at that school as well . When a Daimon attacks Masanori , Chibiusa must take a stand to protect him . 108 `` Everything 's Coming Up Rosey '' `` Usagi Dancing to the Waltz '' `` Usagi no dansu wa warutsu ni notte '' ( うさぎ の ダンス は ワルツ に 乗っ て ) Harume Kosaka Megumi Sugihara Kazuhisa Asai Masahiro Andō September 17 , 1994 July 6 , 2000 An English gentleman named Edwards invites Mamoru and the girls to a fancy party , where Usagi accidentally becomes drunk and the team is forced to confront a fashionable Eudial and her latest Daimon . 109 `` No Turning Back '' `` The Shocking Moment : Everyone 's Identities Revealed '' `` Shōgeki no toki ! Akasareta tagai no shōtai '' ( 衝撃 の 刻 ! 明か され た 互い の 正体 ) Junichi Sato Sukehiro Tomita Yoshiyuki Shikano Kazuya Kuroda September 24 , 1994 July 7 , 2000 Minako becomes jealous that a Daimon has not yet attacked her , and she fears that she does not have a pure heart . After doing as many `` pure '' things as she can , she is attacked , and her pure heart is taken from her . When Eudial nearly gets away , Usagi is forced to reveal her identity to Haruka & Michiru , who in turn , reveal their identities as Sailors Uranus and Neptune to Sailor Moon . 110 `` Destiny 's Arrival '' `` The Death of Uranus and Neptune : The Talisman Appear '' `` Uranusu - tachi no shi ? Tarisuman shutsugen '' ( ウラヌス 達 の 死 ? タリスマン 出現 ) Kunihiko Ikuhara Yoji Enokido Kenichi Tajiri Mari Tominaga October 15 , 1994 July 10 , 2000 Faced with losing her position at Witches 5 , Eudial creates a computer program to find the true owners of the three Talismans , and learns that Haruka and Michiru are the ones who hold two of the Talismans within their own hearts ! They take away Usagi 's brooch , and follow Eudial onto a lonely island . However , the invitation turns out to be a trick , and Michiru 's pure heart is taken , which turns into the mystical Deep Aqua Mirror . Haruka , in order to release her own talisman , shoots herself with Eudial 's gun . Her pure heart transforms into the mystical Space Sword . 111 `` The Purity Chalice '' `` The Holy Grail 's Mystical Power : Moon 's Double Transformation '' `` Seihai no shinpi na chikara ! Mūn nidan henshin '' ( 聖杯 の 神秘 な 力 ! ムーン 二 段 変身 ) Hiroki Shibata Sukehiro Tomita Kazuyuki Hashimoto Ikuko Itō October 22 , 1994 July 11 , 2000 Sailor Pluto appears and reawakens Sailors Uranus and Neptune , and the three talismans resonate with each other ( the third and last being Sailor Pluto 's Garnet Orb ) . The Holy Grail appears , and Sailor Moon grabs it and evolves into Super Sailor Moon , who drives Eudial out . Mimete then kills Eudial by sabotaging her car with snails . 112 `` Show Stoppers '' `` Who is the True Messiah ? Chaos of Light and Darkness '' `` Shin no kyūseishu wa dare ? Hikari to kage no kaosu '' ( 真 の 救世主 は 誰 ? 光 と 影 の カオス ) Noriyo Sasaki Shigeru Yanagawa Minoru Ōkōchi Shinya Hasegawa November 5 , 1994 July 12 , 2000 The girls end up watching the filming of a movie , where Chibiusa encounters Hotaru , whom she quickly befriends . But in the end , she is revealed to be the 12 - year - old daughter of Professor Souichi Tomoe , the possessed leader of the Death Busters . Super Sailor Moon uses her new `` Rainbow Moon Heartache '' attack to defeat the Daimon of the day . 113 `` Rini 's Risky Friendship '' `` A House Filled With Evil Presence : The Beautiful Hotaru 's Secret '' `` Yōki tadayou ie ! Bishōjo Hotaru no himitsu '' ( 妖気 漂う 家 ! 美 少女 ほたる の 秘密 ) Kōnosuke Uda Megumi Sugihara Kenichi Tajiri Taichi Nakamura November 12 , 1994 July 13 , 2000 The three Outer Guardians appear to Tuxedo Mask , and tell him why they can not fight alongside Sailor Moon and the other four Sailor Guardians . Usagi and Chibiusa visit Hotaru , and see a woman who looks like Kaolinite . The Messiah of Silence appears demanding pure hearts . 114 `` Mimet 's Mess '' `` I Love Idols : Mimete 's Dilemma '' `` Aidoru daisuki ! Nayameru Mimetto '' ( アイドル 大好き ! 悩め る ミメット ) Harume Kosaka Yoji Enokido Yoshiyuki Shikano Masahiro Andō November 19 , 1994 July 14 , 2000 Minako and Mimete enter a beauty contest at the same time , with Mimete willing to risk it all to obtain victory . Both fall short of the needed votes to win the idol position , and Mimete decides to just go back to her evil ways . 115 `` The Shadow of Silence '' `` Shadow of Silence : The Pale Glimmer of a Firefly '' `` Chinmoku no kage ! ? Awaki Hotaru hi no yurameki '' ( 沈黙 の 影 ! ? あ わき 蛍火 の ゆらめき ) Takuya Igarashi Yoji Enokido Kenichi Tajiri Katsumi Tamegai November 26 , 1994 July 17 , 2000 Hotaru overexerts herself , and is hospitalized , during which her past without friends is shown , and the reasons for it . Meanwhile , Mimete targets a soap opera actor , during which the attacking Daimon runs near Hotaru , who glows and completely freezes the Daimon . 116 `` Thorny Weather '' `` Sunny Skies After a Storm : A Friendship Dedicated to Hotaru '' `` Arashi nochi hare ! Hotaru ni sasageru yūjō '' ( 嵐 のち 晴れ ! ほたる に 捧げる 友情 ) Yūji Endō Megumi Sugihara Kazuhisa Asai Jouji Yanase December 3 , 1994 July 18 , 2000 Mamoru takes Usagi , Chibiusa and Hotaru to a botanical garden that is tended by Yoshiki Kurebayashi , an old classmate of his . While there , Mimete targets Yoshiki 's pure heart and causes a storm that puts the plants at risk . Sailor Moon defeats the accompanying Daimon , U-Bara . 117 `` Heightened Hazard '' `` Higher and Stronger : A Cheer from Usagi '' `` Yori takaku yori tsuyoku ! Usagi no ōen '' ( より 高く より 強く ! うさぎ の 応援 ) Kunihiko Ikuhara Shigeru Yanagawa Minoru Ōkōchi Taichi Nakamura December 10 , 1994 July 19 , 2000 Hotaru wishes to give a letter to Shun Hayase , a famous athlete . He becomes targeted when she goes to give it to him , and she possessively blasts the enemy with a hugely strong amount of energy . She talks to Shun about being frail and weak , and he thanks her for talking to him . 118 `` It 's In The Cards '' `` The Battle Inside the Demonic Space : The Sailor Guardians ' Gamble '' `` Makū no tatakai ! Sērā Senshi no kake '' ( 魔 空 の 戦い ! セーラー 戦士 の 賭け ) Noriyo Sasaki Katsuyuki Sumisawa Kenichi Tajiri Masahide Yanagisawa December 17 , 1994 July 20 , 2000 Mimete attempts to create a Daimon by herself , throwing the entire Tomoe house into another dimension . The Guardians enter the dimension and are defeated by a game - master Daimon named U-Ikasaman . Hotaru manages to assist in her defeat by the use of a bluff . 119 `` Goodness Eclipsed '' `` The Messiah of Silence Awakens ? Stars of Destiny '' `` Chinmoku no Meshia kakusei ? Unmei no hoshiboshi '' ( 沈黙 の メシア の 覚せい ? 運命 の 星々 ) Harume Kosaka Yoji Enokido Kazuhisa Asai Shinya Hasegawa December 24 , 1994 September 13 , 2000 The Sailor Guardians visit a planetarium , where a man is attacked by a Daimon . The symbol of Saturn appears on Hotaru 's forehead , and she attacks the Daimon . However , Sailor Uranus , Neptune , and Pluto attack Hotaru , since she is both Sailor Saturn , the Sailor Guardian of Silence , Ruin and Birth , and also possessed by the Messiah of Silence . Super Sailor Moon takes the blow to protect Hotaru . 120 `` Next In Line '' `` An Invasion from an Another Dimension : Mystery of Infinity Academy '' `` Ijigen kara no shinryaku ! Mugen Gakuen no nazo '' ( 異 次元 から の 侵略 ! 無限 学園 の 謎 ) Kōnosuke Uda Megumi Sugihara Minoru Ōkōchi Katsumi Tamegai January 7 , 1995 July 21 , 2000 Chibiusa goes to Hotaru 's house to find that Hotaru has disappeared , and is devastated about it . At the same time , the Sailor Guardians investigate Infinity Academy , where Mimete tries one final time to defeat them before Tellu literally pulls the plug on her operation . 121 `` Fiendish Ferns '' `` A Bewitching Flower That Steals Hearts : Tellu , the Third Witch '' `` Kokoro wo ubau yōka ! Daisan no majō Teruru '' ( 心 を 奪う 妖花 ! 第 三 の 魔女 テルル ) Takuya Igarashi Shigeru Yanagawa Kazuyuki Hashimoto Taichi Nakamura January 14 , 1995 July 24 , 2000 Tellu develops a new plan to steal pure hearts , using a plant bred for that purpose . Setsuna realizes it is an alien plant made to steal pure hearts , and warns the Sailor Guardians . Tuxedo Mask causes a mutated plant to destroy Tellu , leading to them tracing her back to the mysterious Infinity Academy . 122 `` The Science Of Love '' `` Believe in Love : Ami , the Kindhearted Guardian '' `` Ai wo shinjite ! Ami , kokoro yasashiki Senshi '' ( 愛 を 信じ て ! 亜美 心 優しき 戦士 ) Yūji Endō Sukehiro Tomita Kenichi Tajiri Hideyuki Motohashi January 21 , 1995 July 25 , 2000 Ami investigates the Infinity Academy , where she meets Viluy , the fourth of the Witches 5 . Viluy attempts to use her nanorobots to steal pure hearts from entrance exam takers . But after her battle , her own microscopic robots turn against her , destroying her in the process . 123 `` Wake Up Call '' `` Shadows of Destruction : the Messiah of Silence Awakens '' `` Hametsu no kage ! Chinmoku no Meshia no mezame '' ( 破滅 の 影 ! 沈黙 の メシア の 目覚め ) Harume Kosaka Megumi Sugihara Minoru Ōkōchi Mari Tominaga January 28 , 1995 July 26 , 2000 The last of the Witches 5 , Cyprine steals pure hearts from Infinity Academy students , and the Sailor Guardians engage Cyprine , and her alter - ego , Ptilol , in a showdown . Meanwhile , Chibiusa is abducted by the recently resurrected Kaolinite , and her pure heart is stolen by Mistress 9 , having taken over Hotaru 's body , which allows her to awaken as the Messiah of Silence . 124 `` Who 's Really Who ? '' `` The Coming Terror of Darkness : Struggle of the Eight Guardians '' `` Semari kuru yami no kyōfu ! Kusen no Hassenshi '' ( 迫り 来る 闇 の 恐怖 ! 苦戦 の 8 戦士 ) Takuya Igarashi Yoji Enokido Kazuyuki Hashimoto Ikuko Itō February 4 , 1995 July 28 , 2000 Mistress 9 uses all remaining Daimon eggs to create a powerful energized shield of evil minions around Infinity Academy . Sailor Moon is abducted , as she is the holder of the Holy Grail . After witnessing Sailor Pluto ultimately sacrificing herself to save them by using her forbidden ability to stop time , Sailor Uranus and Sailor Neptune destroy Germatoid , the Daimon possessing Professor Souichi Tomoe with their talismans , the Deep Aqua Mirror and the Space Sword . 125 `` Darkness , My Old Friend '' `` The Shining Shooting Star : Saturn and the Messiah '' `` Kagayaku ryūsei ! Satān soshite kyūseishu '' ( 輝く 流星 ! サターン そして 救世主 ) Kōnosuke Uda Yoji Enokido Kenichi Tajiri Ikuko Itō February 11 , 1995 July 29 , 2000 Sailor Moon is tricked into giving Mistress 9 the Holy Grail , which she uses to summon Pharaoh 90 to Earth . Hotaru regains consciousness with the assistance of Sailor Moon and her father , destroying Mistress 9 and fully awakening the spirit of Sailor Saturn within herself at last . Sailor Saturn herself returns Chibiusa 's pure heart crystal , and then appears before Sailor Moon to ultimately sacrifice herself and destroy Pharaoh 90 once and for all with her destructive powers , but Sailor Moon gains the help of the other six Sailor Guardians to become Super Sailor Moon once more , and save both Hotaru and Sailor Saturn , who is reborn as a baby immediately afterward . 126 `` Second Chance '' `` A New Life : Parting of the Stars of Destiny '' `` Atarashiki seimei ! Unmei no hoshiboshi betsuri no toki '' ( 新しき 生命 ! 運命 の 星々 別離 の 時 ) Yūji Endō Megumi Sugihara Kazuhisa Asai Taichi Nakamura February 18 , 1995 July 30 , 2000 Haruka and Michiru give the newly reborn infant Hotaru to a still - healing Professor Souichi Tomoe , who is finally free of the Death Busters . However , they viciously attack Sailor Moon , for she protected Hotaru 's body and Sailor Saturn , and risked the whole entire world for one mere life . The Silver Crystal throws them back , and they kneel to her as their leader and leave the city . 127 `` Tough Kindness '' `` A Guardian 's Realization : Strength Lies Within a Pure Heart '' `` Senshi no Jikaku ! Tsuyosa wa Pyua na Kokoro no Naka ni '' ( 戦士 の 自覚 ! 強 さ は 純 な 心 の 中 に ) Harume Kosaka Sukehiro Tomita February 25 , 1995 August 1 , 2000 Chibiusa returns to the future after receiving a letter from her mother , Neo-Queen Serenity . Meanwhile , a final Daimon emerges from the rubble of Infinity Academy , and Chibiusa returns just in time to help defeat it .
  Fact 0.0: The table columns are No., Cloverway Inc. title, original Japanese and Viz Media titles, episode director(s), writer(s), art director(s), animation director(s), original airdate, and English airdate in the United States.
  Fact 0.1: Episode 90 has the Cloverway Inc. title 'Star Struck, Bad Luck', the Viz Media title 'Premonition of the Apocalypse: The Mysterious New Guardians Appear', and the original Japanese title romanized as 'Chikyū hōkai no yokan? Nazo no shin Senshi shutsugen'.
  Fact 0.2: Episode 90 was directed by Junichi Sato, written by Sukehiro Tomita, art directed by Kenichi Tajiri, and animation directed by Masahiro Andō. It originally aired on March 19, 1994 and aired in English in the United States on June 12, 2000.
  Fact 0.3: After experiencing apocalyptic visions, Rei is attacked by a powerful monster called a Daimon.
  Fact 0.4: Sailor Moon is unable to fight the Daimon after it disables her brooch, which reverses her transformation.
  Fact 0.5: The Daimon is defeated by a pair of mysterious new warriors.
  Fact 0.6: The girls despondently realize that a new conflict has begun.
  Fact 0.7: Episode 91 has the Cloverway Inc. title 'Crystal Clear Again', the Viz Media title 'The Rod of Love Is Born: Usagi's New Transformation', and the original Japanese title romanized as 'Ai no roddo tanjō! Usagi no shin henshin'.
  Fact 0.8: Episode 91 was directed by Yūji Endō, written by Shigeru Yanagawa, art directed by Yoshiyuki Shikano, and animation directed by Katsumi Tamegai. It originally aired on March 26, 1994 and aired in English in the United States on June 13, 2000.
  Fact 0.9: While Mamoru strives to cheer Usagi up after her failure in the previous battle, a young girl is the latest target of the Death Busters.
  Fact 0.10: Due to Usagi's inability to transform, the under-powered Sailor Team struggles to fight off the Daimon and is once again forced to rely on the mysterious two warriors.
  Fact 0.11: Once the Sailor Team determines the girl's pure heart is not a talisman, they leave the Sailor Team to fend for themselves.
  Fact 0.12: Tuxedo Mask and Usagi hold hands after the latter's refusal to escape.
  Fact 0.13: Through the power of their love, Usagi's disabled Crystal Star brooch transforms into the Cosmic Heart Compact.
  Fact 0.14: Usagi gains the Spiral Heart Moon Rod and, along with it, the ability to transform and defeat the new enemy.
  Fact 0.15: Episode 92 has the Cloverway Inc. title 'Driving Dangerously', the Viz Media title 'A Handsome Boy? Haruka Tenoh's Secret', and the original Japanese title romanized as 'Suteki na bishōnen? Ten'ō Haruka no himitsu'.
  Fact 0.16: Episode 92 was directed by Kunihiko Ikuhara, written by Katsuyuki Sumisawa, art directed by Kazuhisa Asai, and animation directed by Hisashi Kagawa. It originally aired on April 16, 1994 and aired in English in the United States on June 14, 2000.
  Fact 0.17: Avoiding their study group, Usagi and Minako meet Haruka Tenou, an attractive and supposedly male student of a prestigious local school.
  Fact 0.18: Usagi and Minako both become infatuated with Haruka.
  Fact 0.19: After Minako is defeated in a racing game, Michiru arrives to pick up Haruka.
  Fact 0.20: Michiru and Haruka depart, with Usagi and Minako following behind them soon after.
  Fact 0.21: Once Michiru and Haruka separate at a mechanic's shop, Usagi and Minako take this as their chance.
  Fact 0.22: Haruka tells Usagi and Minako about 'him'-self until a Daimon arrives to attack Kameda, the shop mechanic.
  Fact 0.23: The Daimon starts to escape when Usagi and Minako transform into their Guardian forms to chase it down.
  Fact 0.24: Uranus and Neptune arrive to investigate Kameda's pure heart crystal.
  Fact 0.25: When the crystal turns out not to be a talisman, Uranus and Neptune retreat.
  Fact 0.26: Sailor Moon defeats the Daimon.
  Fact 0.27: Later, the rest of the girls are at Crown Game Center with Usagi and Minako in hopes of seeing the cute boy they had told them about.
  Fact 0.28: Michiru arrives with a blonde girl, whom Usagi and Minako initially fail to realize is Haruka.
  Fact 0.29: Once Usagi and Minako realize the blonde girl is Haruka, Minako falls to the ground in anguish.
  Fact 0.30: Haruka notes that she had never said she was male.
  Fact 0.31: Episode 93 has the Cloverway Inc. title 'Bad Harmony', the Viz Media title 'Usagi's Idol: The Graceful Genius Michiru', and the original Japanese title romanized as 'Usagi no akogare! Yūbi no tensai Michiru'.
  Fact 0.32: Episode 93 was directed by Hiroki Shibata, written by Megumi Sugihara, art directed by Kazuyuki Hashimoto, and animation directed by Taichi Nakamura. It originally aired on April 23, 1994 and aired in English in the United States on June 15, 2000.
  Fact 0.33: When Usagi is concerned that she is not smart enough to be with Mamoru after being scolded by the other girls, she seeks advice from Michiru Kaioh.
  Fact 0.34: Michiru Kaioh suggests that Usagi and Mamoru go to a violin recital.
  Fact 0.35: Episode 94 has the Cloverway Inc. title 'Swept Off Her Feet', the Viz Media title 'Protect the Pure Heart: The Three-Way Battle', and the original Japanese title romanized as 'Pyua na kokoro wo mamore! Teki mikata mittsu tomoe ransen'.
  Fact 0.36: Episode 94 was directed by Noriyo Sasaki, written by Sukehiro Tomita, art directed by Kenichi Tajiri, and animation directed by Ikuko Itō. It originally aired on April 30, 1994 and aired in English in the United States on June 16, 2000.
  Fact 0.37: Usagi's friend, Unazuki, is attacked by a Daimon and has her pure heart crystal stolen.
  Fact 0.38: When Sailor Moon tries to take the crystal back to Unazuki, Uranus and Neptune attempt to steal it in order to claim the Talisman.
  Fact 0.39: Episode 95 has the Cloverway Inc. title 'Blinded By Love's Light', the Viz Media title 'Let Moon Help With Your Love Problems', and the original Japanese title romanized as 'Koi no otasuke wa Mūn ni omakase'.
  Fact 0.40: Episode 95 was directed by Harume Kosaka, written by Shigeru Yanagawa, art directed by Yoshiyuki Shikano, and animation directed by Mari Tominaga. It originally aired on May 7, 1994 and aired in English in the United States on June 19, 2000.
  Fact 0.41: Naru and Umino enter an affection contest in order to prove their strong mutual romantic love for each other.
  Fact 0.42: Naru and Umino find themselves targeted by a Daimon.
  Fact 0.43: Episode 96 has the Cloverway Inc. title 'Lita Borrows Trouble', the Viz Media title 'Coldhearted Uranus: Makoto in Danger', and the original Japanese title romanized as 'Reikoku na Uranusu? Makoto no pinchi'.
  Fact 0.44: Episode 96 was directed by Kōnosuke Uda, written by Yoji Enokido, art directed by Kazuhisa Asai, and animation directed by Masahiro Andō. It originally aired on May 14, 1994 and aired in English in the United States on June 20, 2000.
  Fact 0.45: Makoto is targeted by a Daimon, but manages to temporarily evade it.
  Fact 0.46: Haruka wants to analyze Makoto's pure heart crystal, and befriends Makoto in order to get close to her.
  Fact 0.47: Episode 97 has the Cloverway Inc. title 'Damp Spirits', the Viz Media title 'The Labyrinth of Water: Ami Targeted', and the original Japanese title romanized as 'Mizu no rabarinsu! Nerawareta Ami'.
  Fact 0.48: Episode 97 was directed by Junichi Sato, written by Megumi Sugihara, art directed by Kazuyuki Hashimoto, and animation directed by Katsumi Tamegai. It originally aired on May 21, 1994 and aired in English in the United States on June 21, 2000.
  Fact 0.49: Ami's desire to avoid competition leads her to inadvertently offend Michiru in a swimming race.
  Fact 0.50: Usagi accidentally hurts Ami's feelings.
  Fact 0.51: Meanwhile, a Daimon selects Ami as a target.
  Fact 0.52: Episode 98 has the Cloverway Inc. title 'Friendly Foes', the Viz Media title 'To Save Our Friends: Moon and Uranus Join Forces', and the original Japanese title romanized as 'Tomodachi wo sukue! Mūn Uranusu rengō'.
  Fact 0.53: Episode 98 was directed by Yūji Endō, written by Sukehiro Tomita, art directed by Kenichi Tajiri, and animation directed by Taichi Nakamura. It originally aired on May 28, 1994 and aired in English in the United States on June 22, 2000.
  Fact 0.54: After Sailor Neptune is captured and possibly killed by Kaolinite, Sailor Moon and Sailor Uranus find themselves shackled together.
  Fact 0.55: Sailor Moon and Sailor Uranus are forced to reassess their less-than-friendly attitudes toward each other in order to save Sailor Neptune from further harm.
  Fact 0.56: Episode 99 has the Cloverway Inc. title 'Mixed Emotions', the Viz Media title 'A Man's Kindness: Yuichiro Heartbroken by Rei', and the original Japanese title romanized as 'Otoko no yasashisa! Yūichirō Rei ni shitsuren?'.
  Fact 0.57: Episode 99 was directed by Harume Kosaka, written by Megumi Sugihara, art directed by Yoshiyuki Shikano, and animation directed by Shinya Hasegawa. It originally aired on June 18, 1994 and aired in English in the United States on June 23, 2000.
  Fact 0.58: Worried by Rei's intensive meditation, Yuuichirou becomes concerned that she has fallen deeply head-over-heels in love for another man.
  Fact 0.59: When Yuuichirou sees Rei with Haruka and mistakes Haruka for a man, he is convinced that 'he' is Rei's new romantic suitor.
  Fact 0.60: Episode 100 has the Cloverway Inc. title 'Individual Happiness', the Viz Media title 'I Want to Quit Being a Sailor Guardian: Minako's Dilemma', and the original Japanese title romanized as 'Sērā Senshi wo yametai!? Minako no nayami'.
  Fact 0.61: Episode 100 was directed by Hiroki Shibata, written by Yoji Enokido, art directed by Kazuhisa Asai, and animation directed by Mari Tominaga. It originally aired on June 25, 1994 and aired in English in the United States on June 26, 2000.
  Fact 0.62: Minako is despondent about her double life of being a Sailor Guardian and a normal girl.
  Fact 0.63: Meanwhile, an old friend of Minako's, who she once fell in love with, is attacked by a Daimon.
  Fact 0.64: Minako, as Sailor Venus, must use her skills of volleyball to save the old friend.
  Fact 0.65: Episode 101 has the Cloverway Inc. title 'Birthday Blues, Part 1', the Viz Media title 'Usagi in Tears: a Glass Slipper for My Birthday', and the original Japanese title romanized as 'Usagi namida! Tanjōbi ni garasu no kutsu wo'.
  Fact 0.66: Episode 101 was directed by Yūji Endō, written by Katsuyuki Sumisawa, art directed by Kazuyuki Hashimoto, and animation directed by Masahiro Andō. It originally aired on July 2, 1994 and aired in English in the United States on June 27, 2000.
  Fact 0.67: Usagi is furious when Mamoru forgets her 15th birthday.
  Fact 0.68: When Mamoru buys her an expensive pair of glass slippers to make up for it, he fails to realize that Kaolinite sabotaged the shoes with a Daimon egg to steal her pure heart crystal.
  Fact 0.69: Kaolinite steals Usagi's Cosmic Heart Compact, leaving her unable to transform.
  Fact 0.70: Tuxedo Mask gets captured.
  Fact 0.71: Episode 102 has the Cloverway Inc. title 'Birthday Blues, Part Two', the Viz Media title 'The Stolen Pure Heart: Usagi's Crisis', and the original Japanese title romanized as 'Ubawareta pyua na kokoro! Usagi zettai zetsumei'.
  Fact 0.72: Episode 102 was directed by Noriyo Sasaki, written by Katsuyuki Sumisawa, art directed by Kenichi Tajiri, and animation directed by Taichi Nakamura. It originally aired on July 16, 1994 and aired in English in the United States on June 28, 2000.
  Fact 0.73: With the Cosmic Heart Compact stolen and Tuxedo Mask kidnapped, Usagi and the four other Sailor Guardians, along with Sailors Uranus and Neptune, go to retake the compact.
  Fact 0.74: They go to retake the compact so Usagi can become Sailor Moon again and rescue Tuxedo Mask.
  Fact 0.75: They face Kaolinite in a final showdown at Tokyo Tower.
  Fact 0.76: Episode 103 has the Cloverway Inc. title 'Hello, Sailor Mini Moon', the Viz Media title 'The Arrival of a Tiny Pretty Guardian', and the original Japanese title romanized as 'Yatte kita chiccha na Bishōjo Senshi'.
  Fact 0.77: Episode 103 was directed by Kunihiko Ikuhara, written by Shigeru Yanagawa, art directed by Yoshiyuki Shikano, and animation directed by Kazuya Kuroda. It originally aired on August 6, 1994 and aired in English in the United States on June 29, 2000.
  Fact 0.78: Eudial and the Witches 5 are tasked to take over from Kaolinite's position.
  Fact 0.79: The Witches 5 target a genius drummer recruited to play at a festival for Rei.
  Fact 0.80: The Sailor Guardians cannot defeat the Daimon who attacks the drummer.
  Fact 0.81: Chibiusa arrives as the Guardians' newest ally, Sailor Chibi-Moon.
  Fact 0.82: Episode 104 has the Cloverway Inc. title 'Tainted Tea Party', the Viz Media title 'Making New Friends: Chibi Moon's Adventure', and the original Japanese title romanized as 'Tomodachi wo motomete! Chibi Mūn no katsuyaku'.
  Fact 0.83: Episode 104 was directed by Junichi Sato, written by Megumi Sugihara, art directed by Kazuhisa Asai, and animation directed by Ikuko Itō. It originally aired on August 20, 1994 and aired in English in the United States on June 30, 2000.
  Fact 0.84: Chibiusa has returned from Crystal Tokyo for further training.
  Fact 0.85: Chibiusa attempts to adapt to her new home by making friends her age.
  Fact 0.86: Chibiusa tries to befriend a master who does tea ceremonies, only to become the Death Busters' latest target.
  Fact 0.87: Episode 105 has the Cloverway Inc. title 'People Who Need People', the Viz Media title 'I Want Power: Mako Lost in Doubt', and the original Japanese title romanized as 'Pawā ga hoshī! Mako-chan no mayoi michi'.
  Fact 0.88: Episode 105 was directed by Hiroki Shibata, written by Shigeru Yanagawa, art directed by Yoshiyuki Shikano, and animation directed by Mari Tominaga. It originally aired on August 27, 1994 and aired in English in the United States on July 3, 2000.
  Fact 0.89: Makoto is training in the mountains, trying to become much stronger after being easily defeated by a Daimon before.
  Fact 0.90: Makoto's trainer is attacked, and tells her that nobody can live alone.
  Fact 0.91: Makoto attacks the Daimon and fails, and realizes she needs her best friends in order to win.
  Fact 0.92: Episode 106 has the Cloverway Inc. title 'Related By Destiny', the Viz Media title 'The Bond of Destiny: Uranus's Distant Past', and the original Japanese title romanized as 'Unmei no kizuna! Uranusu no tōi hi'.
  Fact 0.93: Episode 106 was directed by Takuya Igarashi, written by Yoji Enokido, art directed by Kenichi Tajiri, and animation directed by Katsumi Tamegai. It originally aired on September 3, 1994 and aired in English in the United States on July 4, 2000.
  Fact 0.94: Haruka remembers the past when she first met Michiru, who told her that she was destined to become Sailor Uranus.
  Fact 0.95: Haruka initially refused that destiny, for she loved her life too much and was afraid of change.
  Fact 0.96: Haruka was attacked, and Sailor Neptune was injured while defending her.
  Fact 0.97: In her anguish, Haruka took the transformation pen and accepted her destiny as Sailor Uranus, the Sailor Guardian of the Sky.
  Fact 0.98: Episode 107 has the Cloverway Inc. title 'Art Appreciation', the Viz Media title 'Art Is an Explosion of Love: Chibiusa's First Love', and the original Japanese title romanized as 'Geijutsu wa ai no bakuhatsu! Chibiusa no hatsukoi'.
  Fact 0.99: Episode 107 was directed by Yūji Endō, written by Katsuyuki Sumisawa, art directed by Minoru Ōkōchi, and animation directed by Taichi Nakamura. It originally aired on September 10, 1994 and aired in English in the United States on July 5, 2000.
  Fact 0.100: Chibiusa meets a boy named Masanori at her art class and develops a crush on him.
  Fact 0.101: Masanori already has a crush on Michiru, who is taking a class at that school as well.
  Fact 0.102: When a Daimon attacks Masanori, Chibiusa must take a stand to protect him.
  Fact 0.103: Episode 108 has the Cloverway Inc. title 'Everything's Coming Up Rosey', the Viz Media title 'Usagi Dancing to the Waltz', and the original Japanese title romanized as 'Usagi no dansu wa warutsu ni notte'.
  Fact 0.104: Episode 108 was directed by Harume Kosaka, written by Megumi Sugihara, art directed by Kazuhisa Asai, and animation directed by Masahiro Andō. It originally aired on September 17, 1994 and aired in English in the United States on July 6, 2000.
  Fact 0.105: An English gentleman named Edwards invites Mamoru and the girls to a fancy party.
  Fact 0.106: At the party, Usagi accidentally becomes drunk.
  Fact 0.107: At the party, the team is forced to confront a fashionable Eudial and her latest Daimon.
  Fact 0.108: Episode 109 has the Cloverway Inc. title 'No Turning Back', the Viz Media title 'The Shocking Moment: Everyone's Identities Revealed', and the original Japanese title romanized as 'Shōgeki no toki! Akasareta tagai no shōtai'.
  Fact 0.109: Episode 109 was directed by Junichi Sato, written by Sukehiro Tomita, art directed by Yoshiyuki Shikano, and animation directed by Kazuya Kuroda. It originally aired on September 24, 1994 and aired in English in the United States on July 7, 2000.
  Fact 0.110: Minako becomes jealous that a Daimon has not yet attacked her, and she fears that she does not have a pure heart.
  Fact 0.111: After doing as many 'pure' things as she can, Minako is attacked, and her pure heart is taken from her.
  Fact 0.112: When Eudial nearly gets away, Usagi is forced to reveal her identity to Haruka and Michiru.
  Fact 0.113: Haruka and Michiru reveal their identities as Sailors Uranus and Neptune to Sailor Moon.
  Fact 0.114: Episode 110 has the Cloverway Inc. title 'Destiny's Arrival', the Viz Media title 'The Death of Uranus and Neptune: The Talisman Appear', and the original Japanese title romanized as 'Uranusu-tachi no shi? Tarisuman shutsugen'.
  Fact 0.115: Episode 110 was directed by Kunihiko Ikuhara, written by Yoji Enokido, art directed by Kenichi Tajiri, and animation directed by Mari Tominaga. It originally aired on October 15, 1994 and aired in English in the United States on July 10, 2000.
  Fact 0.116: Faced with losing her position at Witches 5, Eudial creates a computer program to find the true owners of the three Talismans.
  Fact 0.117: Eudial learns that Haruka and Michiru are the ones who hold two of the Talismans within their own hearts.
  Fact 0.118: Eudial and her companions take away Usagi's brooch, and follow Eudial onto a lonely island.
  Fact 0.119: The invitation turns out to be a trick, and Michiru's pure heart is taken, which turns into the mystical Deep Aqua Mirror.
  Fact 0.120: Haruka, in order to release her own talisman, shoots herself with Eudial's gun.
  Fact 0.121: Haruka's pure heart transforms into the mystical Space Sword.
  Fact 0.122: Episode 111 has the Cloverway Inc. title 'The Purity Chalice', the Viz Media title 'The Holy Grail's Mystical Power: Moon's Double Transformation', and the original Japanese title romanized as 'Seihai no shinpi na chikara! Mūn nidan henshin'.
  Fact 0.123: Episode 111 was directed by Hiroki Shibata, written by Sukehiro Tomita, art directed by Kazuyuki Hashimoto, and animation directed by Ikuko Itō. It originally aired on October 22, 1994 and aired in English in the United States on July 11, 2000.
  Fact 0.124: Sailor Pluto appears and reawakens Sailors Uranus and Neptune.
  Fact 0.125: The three talismans resonate with each other, the third and last being Sailor Pluto's Garnet Orb.
  Fact 0.126: The Holy Grail appears, and Sailor Moon grabs it and evolves into Super Sailor Moon.
  Fact 0.127: Super Sailor Moon drives Eudial out.
  Fact 0.128: Mimete kills Eudial by sabotaging her car with snails.
  Fact 0.129: Episode 112 has the Cloverway Inc. title 'Show Stoppers', the Viz Media title 'Who is the True Messiah? Chaos of Light and Darkness', and the original Japanese title romanized as 'Shin no kyūseishu wa dare? Hikari to kage no kaosu'.
  Fact 0.130: Episode 112 was directed by Noriyo Sasaki, written by Shigeru Yanagawa, art directed by Minoru Ōkōchi, and animation directed by Shinya Hasegawa. It originally aired on November 5, 1994 and aired in English in the United States on July 12, 2000.
  Fact 0.131: The girls end up watching the filming of a movie, where Chibiusa encounters Hotaru, whom she quickly befriends.
  Fact 0.132: In the end, Chibiusa is revealed to be the 12-year-old daughter of Professor Souichi Tomoe, the possessed leader of the Death Busters.
  Fact 0.133: Super Sailor Moon uses her new 'Rainbow Moon Heartache' attack to defeat the Daimon of the day.
  Fact 0.134: Episode 113 has the Cloverway Inc. title 'Rini's Risky Friendship', the Viz Media title 'A House Filled With Evil Presence: The Beautiful Hotaru's Secret', and the original Japanese title romanized as 'Yōki tadayou ie! Bishōjo Hotaru no himitsu'.
  Fact 0.135: Episode 113 was directed by Kōnosuke Uda, written by Megumi Sugihara, art directed by Kenichi Tajiri, and animation directed by Taichi Nakamura. It originally aired on November 12, 1994 and aired in English in the United States on July 13, 2000.
  Fact 0.136: The three Outer Guardians appear to Tuxedo Mask, and tell him why they cannot fight alongside Sailor Moon and the other four Sailor Guardians.
  Fact 0.137: Usagi and Chibiusa visit Hotaru, and see a woman who looks like Kaolinite.
  Fact 0.138: The Messiah of Silence appears demanding pure hearts.
  Fact 0.139: Episode 114 has the Cloverway Inc. title 'Mimet's Mess', the Viz Media title 'I Love Idols: Mimete's Dilemma', and the original Japanese title romanized as 'Aidoru daisuki! Nayameru Mimetto'.
  Fact 0.140: Episode 114 was directed by Harume Kosaka, written by Yoji Enokido, art directed by Yoshiyuki Shikano, and animation directed by Masahiro Andō. It originally aired on November 19, 1994 and aired in English in the United States on July 14, 2000.
  Fact 0.141: Minako and Mimete enter a beauty contest at the same time, with Mimete willing to risk it all to obtain victory.
  Fact 0.142: Both Minako and Mimete fall short of the needed votes to win the idol position.
  Fact 0.143: Mimete decides to go back to her evil ways.
  Fact 0.144: Episode 115 has the Cloverway Inc. title 'The Shadow of Silence', the Viz Media title 'Shadow of Silence: The Pale Glimmer of a Firefly', and the original Japanese title romanized as 'Chinmoku no kage!? Awaki Hotaru hi no yurameki'.
  Fact 0.145: Episode 115 was directed by Takuya Igarashi, written by Yoji Enokido, art directed by Kenichi Tajiri, and animation directed by Katsumi Tamegai. It originally aired on November 26, 1994 and aired in English in the United States on July 17, 2000.
  Fact 0.146: Hotaru overexerts herself and is hospitalized, during which her past without friends is shown, and the reasons for it.
  Fact 0.147: Meanwhile, Mimete targets a soap opera actor.
  Fact 0.148: During the attack on the actor, the attacking Daimon runs near Hotaru, who glows and completely freezes the Daimon.
  Fact 0.149: Episode 116 has the Cloverway Inc. title 'Thorny Weather', the Viz Media title 'Sunny Skies After a Storm: A Friendship Dedicated to Hotaru', and the original Japanese title romanized as 'Arashi nochi hare! Hotaru ni sasageru yūjō'.
  Fact 0.150: Episode 116 was directed by Yūji Endō, written by Megumi Sugihara, art directed by Kazuhisa Asai, and animation directed by Jouji Yanase. It originally aired on December 3, 1994 and aired in English in the United States on July 18, 2000.
  Fact 0.151: Mamoru takes Usagi, Chibiusa and Hotaru to a botanical garden that is tended by Yoshiki Kurebayashi, an old classmate of Mamoru's.
  Fact 0.152: While there, Mimete targets Yoshiki's pure heart and causes a storm that puts the plants at risk.
  Fact 0.153: Sailor Moon defeats the accompanying Daimon, U-Bara.
  Fact 0.154: Episode 117 has the Cloverway Inc. title 'Heightened Hazard', the Viz Media title 'Higher and Stronger: A Cheer from Usagi', and the original Japanese title romanized as 'Yori takaku yori tsuyoku! Usagi no ōen'.
  Fact 0.155: Episode 117 was directed by Kunihiko Ikuhara, written by Shigeru Yanagawa, art directed by Minoru Ōkōchi, and animation directed by Taichi Nakamura. It originally aired on December 10, 1994 and aired in English in the United States on July 19, 2000.
  Fact 0.156: Hotaru wishes to give a letter to Shun Hayase, a famous athlete.
  Fact 0.157: Shun Hayase becomes targeted when Hotaru goes to give the letter to him.
  Fact 0.158: Hotaru possessively blasts the enemy with a hugely strong amount of energy.
  Fact 0.159: Hotaru talks to Shun about being frail and weak, and he thanks her for talking to him.
  Fact 0.160: Episode 118 has the Cloverway Inc. title 'It's In The Cards', the Viz Media title 'The Battle Inside the Demonic Space: The Sailor Guardians' Gamble', and the original Japanese title romanized as 'Makū no tatakai! Sērā Senshi no kake'.
  Fact 0.161: Episode 118 was directed by Noriyo Sasaki, written by Katsuyuki Sumisawa, art directed by Kenichi Tajiri, and animation directed by Masahide Yanagisawa. It originally aired on December 17, 1994 and aired in English in the United States on July 20, 2000.
  Fact 0.162: Mimete attempts to create a Daimon by herself, throwing the entire Tomoe house into another dimension.
  Fact 0.163: The Guardians enter the dimension and are defeated by a game-master Daimon named U-Ikasaman.
  Fact 0.164: Hotaru manages to assist in U-Ikasaman's defeat by the use of a bluff.
  Fact 0.165: Episode 119 has the Cloverway Inc. title 'Goodness Eclipsed', the Viz Media title 'The Messiah of Silence Awakens? Stars of Destiny', and the original Japanese title romanized as 'Chinmoku no Meshia kakusei? Unmei no hoshiboshi'.
  Fact 0.166: Episode 119 was directed by Harume Kosaka, written by Yoji Enokido, art directed by Kazuhisa Asai, and animation directed by Shinya Hasegawa. It originally aired on December 24, 1994 and aired in English in the United States on September 13, 2000.
  Fact 0.167: The Sailor Guardians visit a planetarium, where a man is attacked by a Daimon.
  Fact 0.168: The symbol of Saturn appears on Hotaru's forehead, and she attacks the Daimon.
  Fact 0.169: Sailor Uranus, Neptune, and Pluto attack Hotaru, since she is both Sailor Saturn, the Sailor Guardian of Silence, Ruin and Birth, and also possessed by the Messiah of Silence.
  Fact 0.170: Super Sailor Moon takes the blow to protect Hotaru.
  Fact 0.171: Episode 120 has the Cloverway Inc. title 'Next In Line', the Viz Media title 'An Invasion from an Another Dimension: Mystery of Infinity Academy', and the original Japanese title romanized as 'Ijigen kara no shinryaku! Mugen Gakuen no nazo'.
  Fact 0.172: Episode 120 was directed by Kōnosuke Uda, written by Megumi Sugihara, art directed by Minoru Ōkōchi, and animation directed by Katsumi Tamegai. It originally aired on January 7, 1995 and aired in English in the United States on July 21, 2000.
  Fact 0.173: Chibiusa goes to Hotaru's house to find that Hotaru has disappeared, and is devastated about it.
  Fact 0.174: At the same time, the Sailor Guardians investigate Infinity Academy.
  Fact 0.175: Mimete tries one final time to defeat the Sailor Guardians before Tellu literally pulls the plug on her operation.
  Fact 0.176: Episode 121 has the Cloverway Inc. title 'Fiendish Ferns', the Viz Media title 'A Bewitching Flower That Steals Hearts: Tellu, the Third Witch', and the original Japanese title romanized as 'Kokoro wo ubau yōka! Daisan no majō Teruru'.
  Fact 0.177: Episode 121 was directed by Takuya Igarashi, written by Shigeru Yanagawa, art directed by Kazuyuki Hashimoto, and animation directed by Taichi Nakamura. It originally aired on January 14, 1995 and aired in English in the United States on July 24, 2000.
  Fact 0.178: Tellu develops a new plan to steal pure hearts, using a plant bred for that purpose.
  Fact 0.179: Setsuna realizes the plant is an alien plant made to steal pure hearts, and warns the Sailor Guardians.
  Fact 0.180: Tuxedo Mask causes a mutated plant to destroy Tellu.
  Fact 0.181: This leads the Sailor Guardians to trace Tellu back to the mysterious Infinity Academy.
  Fact 0.182: Episode 122 has the Cloverway Inc. title 'The Science Of Love', the Viz Media title 'Believe in Love: Ami, the Kindhearted Guardian', and the original Japanese title romanized as 'Ai wo shinjite! Ami, kokoro yasashiki Senshi'.
  Fact 0.183: Episode 122 was directed by Yūji Endō, written by Sukehiro Tomita, art directed by Kenichi Tajiri, and animation directed by Hideyuki Motohashi. It originally aired on January 21, 1995 and aired in English in the United States on July 25, 2000.
  Fact 0.184: Ami investigates the Infinity Academy, where she meets Viluy, the fourth of the Witches 5.
  Fact 0.185: Viluy attempts to use her nanorobots to steal pure hearts from entrance exam takers.
  Fact 0.186: After Viluy's battle, her own microscopic robots turn against her, destroying her in the process.
  Fact 0.187: Episode 123 has the Cloverway Inc. title 'Wake Up Call', the Viz Media title 'Shadows of Destruction: the Messiah of Silence Awakens', and the original Japanese title romanized as 'Hametsu no kage! Chinmoku no Meshia no mezame'.
  Fact 0.188: Episode 123 was directed by Harume Kosaka, written by Megumi Sugihara, art directed by Minoru Ōkōchi, and animation directed by Mari Tominaga. It originally aired on January 28, 1995 and aired in English in the United States on July 26, 2000.
  Fact 0.189: Cyprine is the last of the Witches 5.
  Fact 0.190: Cyprine steals pure hearts from Infinity Academy students.
  Fact 0.191: The Sailor Guardians engage Cyprine, and her alter-ego, Ptilol, in a showdown.
  Fact 0.192: Chibiusa is abducted by the recently resurrected Kaolinite.
  Fact 0.193: Chibiusa's pure heart is stolen by Mistress 9, who has taken over Hotaru's body, which allows her to awaken as the Messiah of Silence.
  Fact 0.194: Episode 124 has the Cloverway Inc. title 'Who's Really Who?', the Viz Media title 'The Coming Terror of Darkness: Struggle of the Eight Guardians', and the original Japanese title romanized as 'Semari kuru yami no kyōfu! Kusen no Hassenshi'.
  Fact 0.195: Episode 124 was directed by Takuya Igarashi, written by Yoji Enokido, art directed by Kazuyuki Hashimoto, and animation directed by Ikuko Itō. It originally aired on February 4, 1995 and aired in English in the United States on July 28, 2000.
  Fact 0.196: Mistress 9 uses all remaining Daimon eggs to create a powerful energized shield of evil minions around Infinity Academy.
  Fact 0.197: Sailor Moon is abducted, as she is the holder of the Holy Grail.
  Fact 0.198: After witnessing Sailor Pluto ultimately sacrificing herself to save them by using her forbidden ability to stop time, Sailor Uranus and Sailor Neptune destroy Germatoid.
  Fact 0.199: Germatoid is the Daimon possessing Professor Souichi Tomoe.
  Fact 0.200: Sailor Uranus and Sailor Neptune destroy Germatoid with their talismans, the Deep Aqua Mirror and the Space Sword.
  Fact 0.201: Episode 125 has the Cloverway Inc. title 'Darkness, My Old Friend', the Viz Media title 'The Shining Shooting Star: Saturn and the Messiah', and the original Japanese title romanized as 'Kagayaku ryūsei! Satān soshite kyūseishu'.
  Fact 0.202: Episode 125 was directed by Kōnosuke Uda, written by Yoji Enokido, art directed by Kenichi Tajiri, and animation directed by Ikuko Itō. It originally aired on February 11, 1995 and aired in English in the United States on July 29, 2000.
  Fact 0.203: Sailor Moon is tricked into giving Mistress 9 the Holy Grail, which she uses to summon Pharaoh 90 to Earth.
  Fact 0.204: Hotaru regains consciousness with the assistance of Sailor Moon and her father, destroying Mistress 9.
  Fact 0.205: Hotaru fully awakens the spirit of Sailor Saturn within herself at last.
  Fact 0.206: Sailor Saturn returns Chibiusa's pure heart crystal.
  Fact 0.207: Sailor Saturn appears before Sailor Moon to ultimately sacrifice herself and destroy Pharaoh 90 once and for all with her destructive powers.
  Fact 0.208: Sailor Moon gains the help of the other six Sailor Guardians to become Super Sailor Moon once more.
  Fact 0.209: Super Sailor Moon saves both Hotaru and Sailor Saturn.
  Fact 0.210: Sailor Saturn is reborn as a baby immediately afterward.
  Fact 0.211: Episode 126 has the Cloverway Inc. title 'Second Chance', the Viz Media title 'A New Life: Parting of the Stars of Destiny', and the original Japanese title romanized as 'Atarashiki seimei! Unmei no hoshiboshi betsuri no toki'.
  Fact 0.212: Episode 126 was directed by Yūji Endō, written by Megumi Sugihara, art directed by Kazuhisa Asai, and animation directed by Taichi Nakamura. It originally aired on February 18, 1995 and aired in English in the United States on July 30, 2000.
  Fact 0.213: Haruka and Michiru give the newly reborn infant Hotaru to a still-healing Professor Souichi Tomoe, who is finally free of the Death Busters.
  Fact 0.214: Haruka and Michiru viciously attack Sailor Moon, for she protected Hotaru's body and Sailor Saturn, and risked the whole entire world for one mere life.
  Fact 0.215: The Silver Crystal throws Haruka and Michiru back.
  Fact 0.216: Haruka and Michiru kneel to Sailor Moon as their leader and leave the city.
  Fact 0.217: Episode 127 has the Cloverway Inc. title 'Tough Kindness', the Viz Media title 'A Guardian's Realization: Strength Lies Within a Pure Heart', and the original Japanese title romanized as 'Senshi no Jikaku! Tsuyosa wa Pyua na Kokoro no Naka ni'.
  Fact 0.218: Episode 127 was directed by Harume Kosaka and written by Sukehiro Tomita. It originally aired on February 25, 1995 and aired in English in the United States on August 1, 2000.
  Fact 0.219: Chibiusa returns to the future after receiving a letter from her mother, Neo-Queen Serenity.
  Fact 0.220: Meanwhile, a final Daimon emerges from the rubble of Infinity Academy.
  Fact 0.221: Chibiusa returns just in time to help defeat the final Daimon.

Retrieved items:
<item 1>
*Sailor Moon S* (38 episodes, 1994–95) follows the Death Busters, who target girls’ Pure Heart Crystals while seeking three Talismans and the Holy Grail.
Haruka/Uranus and Michiru/Neptune, later joined by Pluto, investigate independently; their Talismans summon the Grail, empowering Sailor Moon as Super Sailor
Moon. The Witches 5—including Eudial, Mimete, and Viluy—serve the Death
</item 1>

<item 2>
After two mysterious warriors defeat a Daimon, the Sailor Guardians realize a new conflict has begun. Usagi struggles to transform as the Death Busters target
girls’ pure hearts; Uranus and Neptune investigate but leave when the crystals prove not to be talismans, forcing Sailor Moon to defeat the Daimon. Meanwhile,
Usagi and Minako follow Haruka and Michiru, misunderstand Haruka’s gender, and later learn she never claimed to be a boy.
</item 2>

<item 3>
In *Sailor Moon S*, Usagi and Tuxedo Mask’s love transforms her damaged Crystal Star Brooch into the Cosmic Heart Compact and grants her the Spiral Heart Moon
Rod. Usagi and Minako meet Haruka Tenoh and become infatuated with the handsome student. Minako later saves an old love from a Daimon using her volleyball
skills. Kaolinite sabotages Mamoru’s birthday gift to Usagi, steals Usagi’s compact, and captures Tuxedo
</item 3>

<item 4>
List of Sailor Moon S episodes
タリスマン 出現 ) Kunihiko Ikuhara Yoji Enokido Kenichi Tajiri Mari Tominaga October 15 , 1994 July 10 , 2000 Faced with losing her position at Witches 5 , Eudial
creates a computer program to find the true owners of the three Talismans , and learns that Haruka and Michiru are the ones who hold two of the Talismans within
their own hearts ! They take away Usagi 's brooch , and follow Eudial onto a lonely island . However , the invitation turns out to be a trick , and Michiru 's
pure heart is taken , which turns into the mystical Deep Aqua Mirror . Haruka , in order to release her own talisman , shoots herself with Eudial 's gun . Her
pure heart transforms into the mystical Space Sword . 111 `` The Purity Chalice '' `` The Holy Grail 's Mystical Power : Moon 's Double Transformation '' ``
Seihai no shinpi na chikara !
</item 4>

<item 5>
List of Sailor Moon S episodes
The Daimon is defeated by a pair of mysterious new warriors , leaving the girls to despondently realize that a new conflict has begun . 91 `` Crystal Clear
Again '' `` The Rod of Love Is Born : Usagi 's New Transformation '' `` Ai no roddo tanjō ! Usagi no shin henshin '' ( 愛 の ロッド 誕生 ! うさぎ の 新 変身 ) Yūji Endō
Shigeru Yanagawa Yoshiyuki Shikano Katsumi Tamegai March 26 , 1994 June 13 , 2000 While Mamoru strives to cheer Usagi up after her failure in the previous
battle , a young girl is the latest target of the Death Busters . Due to Usagi 's inability to transform , the under - powered Sailor Team struggles to fight
off the Daimon and are once again forced to rely on the mysterious two warriors . Once they determine the girl 's pure heart is not a talisman , they leave the
Sailor Team to fend for themselves .
</item 5>

<item 6>
List of Sailor Moon S episodes
Meanwhile , Tuxedo Mask and Usagi hold hands after the latter 's refusal to escape , and through the power of their love , Usagi 's disabled Crystal Star brooch
transforms into the Cosmic Heart Compact , and she gains the Spiral Heart Moon Rod and , along with it , the ability to transform and defeat the new enemy . 92
`` Driving Dangerously '' `` A Handsome Boy ? Haruka Tenoh 's Secret '' `` Suteki na bishōnen ? Ten'ō Haruka no himitsu '' ( 素敵 な 美少年 ? 天王 はるか の 秘密 ) Kunihiko
Ikuhara Katsuyuki Sumisawa Kazuhisa Asai Hisashi Kagawa April 16 , 1994 June 14 , 2000 Avoiding their study group , Usagi and Minako meet Haruka Tenou , an
attractive and supposedly male student of a prestigious local school , with whom they both become infatuated .
</item 6>

<item 7>
List of Sailor Moon S episodes
Satān soshite kyūseishu '' ( 輝く 流星 ! サターン そして 救世主 ) Kōnosuke Uda Yoji Enokido Kenichi Tajiri Ikuko Itō February 11 , 1995 July 29 , 2000 Sailor Moon is tricked
into giving Mistress 9 the Holy Grail , which she uses to summon Pharaoh 90 to Earth . Hotaru regains consciousness with the assistance of Sailor Moon and her
father , destroying Mistress 9 and fully awakening the spirit of Sailor Saturn within herself at last . Sailor Saturn herself returns Chibiusa 's pure heart
crystal , and then appears before Sailor Moon to ultimately sacrifice herself and destroy Pharaoh 90 once and for all with her destructive powers , but Sailor
Moon gains the help of the other six Sailor Guardians to become Super Sailor Moon once more , and save both Hotaru and Sailor Saturn , who is reborn as a baby
immediately afterward . 126 `` Second Chance '' `` A New Life : Parting of the Stars of Destiny '' `` Atarashiki seimei !
</item 7>

<item 8>
The excerpts cover several *Sailor Moon S* episodes. Usagi doubts she is worthy of Mamoru and takes Michiru’s advice to attend a violin recital with him. The
Outer Guardians explain why they cannot fight alongside the others; Usagi and Chibiusa visit Hotaru, where the Messiah of Silence appears. Minako and Mimete
compete in a beauty contest. Mimete later targets a botanist’s Pure Heart, endangering his plants, and Hotaru hopes
</item 8>

<item 9>
List of Sailor Moon S episodes
He becomes targeted when she goes to give it to him , and she possessively blasts the enemy with a hugely strong amount of energy . She talks to Shun about
being frail and weak , and he thanks her for talking to him . 118 `` It 's In The Cards '' `` The Battle Inside the Demonic Space : The Sailor Guardians '
Gamble '' `` Makū no tatakai ! Sērā Senshi no kake '' ( 魔 空 の 戦い ! セーラー 戦士 の 賭け ) Noriyo Sasaki Katsuyuki Sumisawa Kenichi Tajiri Masahide Yanagisawa December
17 , 1994 July 20 , 2000 Mimete attempts to create a Daimon by herself , throwing the entire Tomoe house into another dimension . The Guardians enter the
dimension and are defeated by a game - master Daimon named U-Ikasaman . Hotaru manages to assist in her defeat by the use of a bluff . 119 `` Goodness Eclipsed
'' `` The Messiah of Silence Awakens ?
</item 9>

<item 10>
Sailor Pluto reawakens Uranus and Neptune, and their three talismans summon the Holy Grail. Sailor Moon becomes Super Sailor Moon and drives off Eudial, whom
Mimete later kills by sabotaging her car. Later, Mistress 9 steals the Grail and summons Pharaoh 90. Hotaru destroys Mistress 9 and awakens as Sailor Saturn,
who sacrifices herself to defeat Pharaoh 90. With the other Guardians’ help, Super Sailor Moon saves Hotaru
</item 10>
=== END ITEM id=528b98aecf08fe69 ===

=== ITEM id=58defafbc7ab37a1 ===
Gold sentences and facts:
Sentence 0: We admitted we were powerless over alcohol -- that our lives had become unmanageable . Came to believe that a Power greater than ourselves could restore us to sanity . Made a decision to turn our will and our lives over to the care of God as we understood Him . Made a searching and fearless moral inventory of ourselves . Admitted to God , to ourselves , and to another human being the exact nature of our wrongs . Were entirely ready to have God remove all these defects of character . Humbly asked Him to remove our shortcomings . Made a list of all persons we had harmed , and became willing to make amends to them all . Made direct amends to such people wherever possible , except when to do so would injure them or others . Continued to take personal inventory , and when we were wrong , promptly admitted it . Sought through prayer and meditation to improve our conscious contact with God as we understood Him , praying only for knowledge of His will for us and the power to carry that out . Having had a spiritual awakening as the result of these steps , we tried to carry this message to alcoholics , and to practice these principles in all our affairs .
  Fact 0.0: We admitted we were powerless over alcohol.
  Fact 0.1: We admitted our lives had become unmanageable.
  Fact 0.2: We came to believe that a Power greater than ourselves could restore us to sanity.
  Fact 0.3: We made a decision to turn our will and our lives over to the care of God as we understood Him.
  Fact 0.4: We made a searching and fearless moral inventory of ourselves.
  Fact 0.5: We admitted to God, to ourselves, and to another human being the exact nature of our wrongs.
  Fact 0.6: We were entirely ready to have God remove all these defects of character.
  Fact 0.7: We humbly asked God to remove our shortcomings.
  Fact 0.8: We made a list of all persons we had harmed.
  Fact 0.9: We became willing to make amends to all persons we had harmed.
  Fact 0.10: We made direct amends to such people wherever possible, except when to do so would injure them or others.
  Fact 0.11: We continued to take personal inventory.
  Fact 0.12: When we were wrong, we promptly admitted it.
  Fact 0.13: We sought through prayer and meditation to improve our conscious contact with God as we understood Him.
  Fact 0.14: We prayed only for knowledge of God's will for us.
  Fact 0.15: We prayed only for the power to carry out God's will for us.
  Fact 0.16: Having had a spiritual awakening as the result of these steps, we tried to carry this message to alcoholics.
  Fact 0.17: We tried to practice these principles in all our affairs.
Sentence 1: We admitted we were powerless over alcohol -- that our lives had become unmanageable .
  Fact 1.0: We admitted we were powerless over alcohol.
  Fact 1.1: We admitted our lives had become unmanageable.

Retrieved items:
<item 1>
Twelve-step program
Twelve - step program - wikipedia Twelve - step program Jump to : navigation , search A twelve - step program is a set of guiding principles outlining a course
of action for recovery from addiction , compulsion , or other behavioral problems . Originally proposed by Alcoholics Anonymous ( AA ) as a method of recovery
from alcoholism , the Twelve Steps were first published in the 1939 book Alcoholics Anonymous : The Story of How More Than One Hundred Men Have Recovered from
Alcoholism . The method was adapted and became the foundation of other twelve - step programs . As summarized by the American Psychological Association , the
process involves the following : admitting that one can not control one 's alcoholism , addiction or compulsion ; recognizing a higher power that can give
strength ; examining past errors with the help of a sponsor ( experienced member ) ; making amends for these errors ; learning to live a new life with a new
code of behavior ; helping others who suffer from the same alcoholism , addictions or compulsions .
</item 1>

<item 2>
Twelve-step program
`` Twelve - step recovery and community service '' . Health & Social Work . 28 ( 2 ) : 137 -- 145 . doi : 10.1093 / hsw / 28.2. 137 . CS1 maint : Multiple names
: authors list ( link ) Lile , B. ( 2003 ) . `` Twelve step programs : An update '' . Addictive Disorders & Their Treatment . 2 ( 1 ) : 19 -- 24 . Morgenstern ,
J. , Bux , D. , Labouvie , E. , Blanchard , K.A. , & Morgan , T.J. ( November 2002 ) . `` Examining mechanisms of action in 12 - step treatment : The role of 12
- step cognitions '' . Journal of Studies on Alcohol . 63 ( 6 ) : 665 -- 672 . PMID 12529066 . CS1 maint : Multiple names : authors list ( link ) Sheehan , T. (
2004 ) . `` Twelve Step Facilitation : A Necessary Treatment for Offenders '' .
</item 2>

<item 3>
Twelve-step program
Contents ( hide ) 1 Overview 2 History 3 Twelve Steps 4 Twelve Traditions 5 Process 6 Sponsorship 7 Effectiveness 8 Criticism 8.1 Confidentiality 8.2 Cultural
identity 9 See also 10 References 11 Further reading 11.1 Scholarly publications Overview ( edit ) Twelve - step methods have been adapted to address a wide
range of alcoholism , substance - abuse and dependency problems . Over 200 self - help organizations -- often known as fellowships -- with a worldwide
membership of millions -- now employ twelve - step principles for recovery . Narcotics Anonymous was formed by addicts who did not relate to the specifics of
alcohol dependency . Demographic preferences related to the addicts ' drug of choice has led to the creation of Cocaine Anonymous , Crystal Meth Anonymous and
Marijuana Anonymous . Behavioral issues such as compulsion for , and / or addiction to , gambling , crime , food , sex , hoarding , debting and work are
addressed in fellowships such as Gamblers Anonymous , Overeaters Anonymous , Sexaholics Anonymous and Debtors Anonymous .
</item 3>

<item 4>
Twelve-step program
Made a searching and fearless moral inventory of ourselves . Admitted to God , to ourselves , and to another human being the exact nature of our wrongs . Were
entirely ready to have God remove all these defects of character . Humbly asked Him to remove our shortcomings . Made a list of all persons we had harmed , and
became willing to make amends to them all . Made direct amends to such people wherever possible , except when to do so would injure them or others . Continued
to take personal inventory , and when we were wrong , promptly admitted it . Sought through prayer and meditation to improve our conscious contact with God as
we understood Him , praying only for knowledge of His will for us and the power to carry that out . Having had a spiritual awakening as the result of these
steps , we tried to carry this message to alcoholics , and to practice these principles in all our affairs . In some cases , where other twelve - step groups
have adapted the AA steps as guiding principles , step one is uniquely different for each organization , these are sometimes altered to emphasize principles
important to those particular fellowships , or to remove gender - biased language .
</item 4>

<item 5>
Twelve-step program
ISSN 1552 - 6879 . OCLC 1783135 . Jump up ^ Alcoholics Anonymous ( 2001 - 06 - 01 ) . `` Chapter 2 : There Is a Solution '' . Alcoholics Anonymous ( PDF ) ( 4th
ed . ) . Alcoholics Anonymous World Services . p. 21 . ISBN 1893007162 . OCLC 32014950 . At some stage of his drinking career , he begins to lose all control of
his liquor consumption , once he starts to drink . Jump up ^ Alcoholics Anonymous ( 2001 - 06 - 01 ) . `` Chapter 2 : There Is a Solution '' . Alcoholics
Anonymous ( PDF ) ( 4th ed . ) . Alcoholics Anonymous World Services . p. 21 . ISBN 1893007162 . OCLC 32014950 . These observations would be academic and
pointless if ( he ) never took the first drink , thereby setting the terrible cycle in motion .
</item 5>

<item 6>
Twelve-step program
Retrieved September 28 , 2012 . Jump up ^ Bill W. ( April 1946 ) . `` Our A.A. Experience Has Taught Us That : '' . The A.A. Grapevine . Alcoholics Anonymous .
2 ( 11 ) . ISSN 0362 - 2584 . OCLC 50379271 . ^ Jump up to : Kurtz LF , Chambon A ( 1987 ) . `` Comparison of self - help groups for mental health '' . Health &
Social Work . 12 ( 4 ) : 275 -- 83 . PMID 3679015 . ^ Jump up to : Ronel , Natti ( 2000 ) . `` From Self - Help to Professional Care : An Enhanced Application
of the 12 - Step Program '' . The Journal of Applied Behavioral Science . 36 ( 1 ) : 108 -- 122 . doi : 10.1177 / 0021886300361006 . ISSN 1552 - 6879 . OCLC
1783135 .
</item 6>

<item 7>
Twelve-step program
As AA chapters were increasing in number during the 1930s and 1940s , the guiding principles were gradually defined as the Twelve Traditions . A singleness of
purpose emerged as Tradition Five : `` Each group has but one primary purpose -- to carry its message to the alcoholic who still suffers '' . Consequently ,
drug addicts who do not suffer from the specifics of alcoholism involved in AA hoping for recovery technically are not welcome in `` closed '' meetings unless
they have a desire to stop drinking alcohol . The principles of AA have been used to form many numbers of other fellowships specifically designed for those
recovering from various pathologies ; each emphasizes recovery from the specific malady which brought the sufferer into the fellowship . Twelve steps ( edit )
For more details on this topic , see List of Twelve Step alternate wordings . The following are the original twelve steps as published by Alcoholics Anonymous :
We admitted we were powerless over alcohol -- that our lives had become unmanageable . Came to believe that a Power greater than ourselves could restore us to
sanity . Made a decision to turn our will and our lives over to the care of God as we understood Him .
</item 7>

<item 8>
Twelve-step program
. Alcoholics Anonymous World Services . ISBN 1 - 893007 - 16 - 2 . OCLC 32014950 . Jump up ^ `` World Service Board of Trustees Bulletin # 13 : Some thoughts
regarding our relationship to Alcoholics Anonymous '' . Narcotics Anonymous World Services . February 23 , 2007 . Archived from the original on October 6 , 2007
. Retrieved October 7 , 2007 . Jump up ^ `` NA History Chronology '' . January 13 , 2006 . Archived from the original on March 3 , 2009 . Retrieved October 7 ,
2007 . 1953 : ... this committee met regularly ... From the beginning ... 12 Steps ... 12 Traditions ... All Steps had ' We ' Jump up ^ `` The 12 Steps of
Recovery '' . Archived from the original on 2012 - 11 - 05 . Retrieved August 28 , 2010 . Jump up ^ `` The Twelve Steps of Marijuana Anonymous '' . Retrieved
September 28 , 2012 . Jump up ^ Bill W.
</item 8>

<item 9>
Twelve-step program
The A.A. Grapevine . Alcoholics Anonymous . 6 ( 6 ) . November 1949 . ISSN 0362 - 2584 . OCLC 50379271 . Jump up ^ Twelve Steps and Twelve Traditions . Hazelden
. February 2002 . ISBN 0 - 916856 - 01 - 1 . OCLC 13572433 . Jump up ^ `` For Anyone New Coming to A.A. ; For Anyone Referring People to A.A '' . Alcoholics
Anonymous World Services , Inc . Archived from the original on 2008 - 01 - 17 . Retrieved June 15 , 2006 . Jump up ^ Vaillant , George E. ( 2002 ) . ``
Singleness of Purpose '' ( PDF ) . About AA : A Newsletter for Professionals ( Fall / Winter ) . Jump up ^ Bill W. ( June 2001 ) . `` Chapter 5 : How It Works
'' . Alcoholics Anonymous ( PDF ) ( 4th ed . ) . Alcoholics Anonymous World Services .
</item 9>

<item 10>
Twelve-step program
137.4. 409 . PMID 6987904 . Jump up ^ Alcoholics Anonymous ( 1976 - 06 - 01 ) . `` Appendix II . Spiritual Experience '' . Alcoholics Anonymous . Alcoholics
Anonymous World Services . ISBN 0916856593 . OCLC 32014950 . Archived from the original on 2009 - 04 - 27 . Jump up ^ Hayes , Terrell ( February 2000 ) . ``
Stigmatizing Indebtedness : Implications for Labeling Theory '' . Symbolic Interaction. 23 ( 1 ) : 29 -- 46 . doi : 10.1525 / si. 2000.23. 1.29 . ^ Jump up to :
`` Sponsorship Q&A ( pamphlet ) '' ( PDF ) . New York : Alcoholics Anonymous World Services . Jump up ^ `` NYCMA : What is a Sponsor ? '' . New York Crystal
Meth Anonymous Intergroup . Archived from the original on 2016 - 04 - 04 . Retrieved October 8 , 2007 .
</item 10>
=== END ITEM id=58defafbc7ab37a1 ===

=== ITEM id=0a347f33c64441f4 ===
Gold sentences and facts:
Sentence 0: The culture of Australia is a Western culture , derived primarily from Britain but also influenced by the unique geography of Australia , the diverse input of Aboriginal , Torres Strait Islander and other Australian people . The British colonization of Australia began in 1788 , and waves of multi-ethnic migration followed . Evidence of a significant Anglo - Celtic heritage includes the predominance of the English language , the existence of a democratic system of government drawing upon the British traditions of Westminster Government , Parliamentarianism and constitutional monarchy , American constitutionalist and federalist traditions , Christianity as the dominant religion , and the popularity of sports originating in ( or influenced by ) the British Isles . Australian culture has diverged significantly since British settlement .
  Fact 0.0: The culture of Australia is a Western culture.
  Fact 0.1: The culture of Australia is derived primarily from Britain.
  Fact 0.2: The culture of Australia is also influenced by the unique geography of Australia.
  Fact 0.3: The culture of Australia is influenced by the diverse input of Aboriginal, Torres Strait Islander and other Australian people.
  Fact 0.4: British colonization of Australia began in 1788.
  Fact 0.5: Waves of multi-ethnic migration followed the British colonization of Australia.
  Fact 0.6: Evidence of a significant Anglo-Celtic heritage in Australia includes the predominance of the English language.
  Fact 0.7: Evidence of Anglo-Celtic heritage in Australia includes a democratic system of government drawing upon the British traditions of Westminster Government, Parliamentarianism and constitutional monarchy.
  Fact 0.8: Evidence of Anglo-Celtic heritage in Australia includes American constitutionalist and federalist traditions.
  Fact 0.9: Evidence of Anglo-Celtic heritage in Australia includes Christianity as the dominant religion.
  Fact 0.10: Evidence of Anglo-Celtic heritage in Australia includes the popularity of sports originating in (or influenced by) the British Isles.
  Fact 0.11: Australian culture has diverged significantly since British settlement.

Retrieved items:
<item 1>
Culture of Australia
By 2010 , the post-war immigration program had received more than 6.5 million migrants from every continent . The population tripled in the six decades to
around 21 million in 2010 , including people originating from 200 countries . More than 43 per cent of Australians were either born overseas or have one parent
who was born overseas . The population is highly urbanised , with more than 75 % of Australians living in urban centres , largely along the coast . Contemporary
Australia is a pluralistic society , rooted in liberal democratic traditions and espousing informality and egalitarianism as key societal values . While
strongly influenced by Anglo - Celtic origins , the culture of Australia has also been shaped by multi-ethnic migration which has influenced all aspects of
Australian life , including business , the arts , cuisine , sense of humour and sporting tastes . Contemporary Australia is also a culture that is profoundly
influenced by global movements of meaning and communication , including advertising culture . In turn , globalizing corporations from Holden to Exxon have
attempted to associate their brand with Australian cultural identity . This process intensified from the 1970s onwards .
</item 1>

<item 2>
Culture of Australia
Culture of Australia - Wikipedia Culture of Australia Jump to : navigation , search Part of a series on the Culture of Australia History People ( show )
Australian people African Australian Albanian Australian American Australian Anglo - Celtic Australian Asian Australian Assyrian Australian Bosnian Australian
Chinese Australian Cornish Australian English Australian European Australian Filipino Australian Greek Australian Indian Australian Indigenous Australian Irish
Australian Italian Australian Japanese Australian Jewish Australian Korean Australian Latin American Australian Lebanese Australian Macedonian Australian
Pakistani Australian Scottish Australian Serbian Australian South African Australian Sudanese Australian Taiwanese Australian Welsh Australian Languages
Traditions ( show ) Humour Mythology and folklore ( show ) Mythology folklore Cuisine Festivals Religion ( show ) Aboriginal Christianity ( Roman Catholic
Anglican Uniting Church Presbyterian Baptist Lutheran Coptic Orthodox more ) Buddhism Islam Sikhism Judaism Hinduism Bahá'í Faith Irreligion Art ( show )
Architecture Literature ( show ) Comic books Music and performing arts ( show ) Music Performing arts Media ( show ) Radio Television Cinema Sport Monuments (
show ) World Heritage Sites Symbols ( show ) Flag Coat of arms Australia portal The culture of Australia is a Western culture , derived primarily from Britain
but also influenced by the unique geography of Australia , the diverse input of Aboriginal , Torres Strait Islander and other Australian people .
</item 2>

<item 3>
Culture of Australia
These wars profoundly affected Australia 's sense of nationhood and a proud military legend developed around the spirit of Australia 's ANZAC troops , who came
to symbolise the virtues of mateship , courage and endurance for the nation . The Australian colonies had a period of extensive multi-ethnic immigration during
the Australian gold rushes of the latter half of the 19th century , but following Federation in 1901 , the Parliament instigated the White Australia Policy that
gave preference to British migrants and ensured that Australia remained a predominantly Anglo - Celtic society until well into the 20th Century . The post-World
War II immigration program saw the policy dismantled by successive governments , permitting large numbers of Southern European , and later Asian and Middle
Eastern migrants to arrive . The Menzies Government ( 1949 - 1966 ) and Holt Government dismantled the legal barriers to multi-ethnic immigration and by the
1970s , the Whitlam and Fraser Governments were promoting multiculturalism . Countries of birth of Australian estimated resident population , 2006 Some States
and Territories of Australia retained discriminatory laws relating to voting rights for Aborigines into the 1960s , at which point full legal equality was
established .
</item 3>

<item 4>
Culture of Australia
The British colonization of Australia began in 1788 , and waves of multi-ethnic migration followed . Evidence of a significant Anglo - Celtic heritage includes
the predominance of the English language , the existence of a democratic system of government drawing upon the British traditions of Westminster Government ,
Parliamentarianism and constitutional monarchy , American constitutionalist and federalist traditions , Christianity as the dominant religion , and the
popularity of sports originating in ( or influenced by ) the British Isles . Australian culture has diverged significantly since British settlement . Aboriginal
people are believed to have arrived as early as 60,000 years ago , and evidence of Aboriginal art in Australia dates back at least 30,000 years . Several states
and territories had their origins as penal colonies , with the first British convicts arriving at Sydney Cove in 1788 . Stories of outlaws like the bushranger
Ned Kelly have endured in Australian music , cinema and literature . The Australian gold rushes from the 1850s brought wealth as well as new social tensions to
Australia , including the miners ' Eureka Stockade rebellion . The colonies established elected parliaments and rights for workers and women before most other
Western nations .
</item 4>

<item 5>
Culture of Australia
A 1967 referendum to include all Aborigines in the national electoral roll census was overwhelmingly approved by voters . In 1984 , a group of Pintupi people
who were living a traditional hunter - gatherer desert - dwelling life were tracked down in the Gibson Desert and brought into a settlement . They are believed
to have been the last uncontacted tribe . While the British cultural influence remained strong into the 21st century , other influences became increasingly
important . The Hawaiian sport of surfing was adopted in Australia where a beach culture and the locally developed surf lifesaving movement was already
burgeoning in the early 20th century . American pop culture and cinema were embraced in the 20th century , with country music and later rock and roll sweeping
Australia , aided by the new technology of television and a host of American content . The 1956 Melbourne Olympics announced a confident , prosperous post-war
nation , and new cultural icons like Australian country music star Slim Dusty and dadaist Barry Humphries expressed a uniquely Australian identity . Australia
's contemporary immigration program has two components : a program for skilled and family migrants and a humanitarian program for refugees and asylum seekers .
</item 5>

<item 6>
Culture of Australia
Hyde Park Barracks , Sydney Convict architecture at Port Arthur , Tasmania Interior of St Mary 's Cathedral , Sydney The Royal Exhibition Building , Melbourne
Birdsville Hotel , an Australian pub in outback Queensland Parliament House , Canberra Federation Bungalow in Killara , Sydney A typical Queenslander house in
Brisbane Visual arts ( edit ) Main article : Visual arts of Australia See also : Indigenous Australian art Bradshaws in the Kimberley region of Western
Australia Sunbaker ( 1937 ) , an iconic photograph by Max Dupain Aboriginal rock art is the oldest continuous art tradition in the world , dating as far back as
60,000 years . From the Bradshaw and Wondjina imagery in the Kimberley to the Sydney rock engravings , it is spread across hundreds of thousands of sites ,
making Australia the richest continent in terms of prehistoric art . 19th century Indigenous activist William Barak painted ceremonial scenes , such as
corroborees . The Hermannsburg School , led by Albert Namatjira , received national fame in the 1950s for their desert watercolours . Leading critic Robert
Hughes saw contemporary Indigenous art as `` the last great art movement of the 20th century '' .
</item 6>

<item 7>
Culture of Australia
1 Indigenous music 5.6. 2 Folk music and national songs 5.6. 3 Classical music 5.6. 4 Popular music 5.7 Dance 5.8 Television 6 Religion 7 Public holidays 8
Cuisine 8.1 Beverages 9 Clothing and apparel 10 Sport 10.1 Cricket 10.2 Football codes 10.3 Water sports 10.4 Other sports 11 Folklore 12 Attitudes , beliefs
and stereotypes 13 References Development of Australian culture ( edit ) Main article : History of Australia A Luritja man demonstrating method of attack with
boomerang under cover of shield ( 1920 ) The oldest surviving cultural traditions in Australia -- and some of the oldest surviving cultural traditions on earth
-- are those of Australia 's Aboriginal and Torres Strait Islander peoples . Their ancestors have inhabited Australia for between 40,000 and 60,000 years ,
living a hunter - gatherer lifestyle . In 2006 , the Indigenous population was estimated at 517,000 people , or 2.5 per cent of the total population .
</item 7>

<item 8>
Culture of Australia
They established agricultural industries producing more familiar Western style produce . Queensland and New South Wales became Australia 's main beef cattle
producers , while dairy cattle farming is found in the southern states , predominantly in Victoria . Wheat and other grain crops are spread fairly evenly
throughout the mainland states . Sugar cane is also a major crop in Queensland and New South Wales . Fruit and vegetables are grown throughout Australia . ``
Meat and three veg '' , fish and chips , and the Australian meat pie continue to represent traditional meals for many Australians . The post-World War II
multicultural immigration program brought new flavours and influences , with waves of immigrants from Greece , Italy , Vietnam , China , and elsewhere bringing
about diversification of the typical diet consumed . Australia 's 11 million square kilometre fishing zone is the third largest in the world and allows for easy
access to seafood which significantly influences Australian cuisine . Clean ocean environments produce high quality seafoods .
</item 8>

<item 9>
Culture of Australia
The colonies established elected parliaments and rights for workers and women before most other Western nations . Federation in 1901 evidenced a growing sense
of national identity that had developed over the latter half of the 19th century , as seen in the works of the Heidelberg School painters and writers like Banjo
Paterson , Henry Lawson and Dorothea Mackellar . The World Wars profoundly altered Australia 's sense of identity , with World War I introducing the ANZAC
legend , and World War II seeing a reorientation from Britain to the United States as the nation 's foremost major ally . After the second war , 6.5 million
migrants from 200 nations brought immense new diversity , and Australians grew increasingly aware of their proximity to Asia . Over time , the diverse food ,
lifestyle and cultural practices of immigrants have been absorbed into mainstream Australian culture . Contents ( hide ) 1 Development of Australian culture 2
Symbols 3 Language 4 Humour 5 Arts 5.1 Literature 5.2 Theatre 5.3 Architecture 5.4 Visual arts 5.5 Cinema 5.6 Music 5.6. 1 Indigenous music 5.6.
</item 9>

<item 10>
Culture of Australia
Most Aboriginal Australians have a belief system based on the Dreaming , or Dreamtime , which refers both to a time when ancestral spirits created land and
culture , and to the knowledge and practices that define individual and community responsibilities and identity . Conflict and reconciliation between indigenous
and non-indigenous Australians has been a source of much art and literature in Australia , and ancient Aboriginal artistic styles and iconic inventions such as
the boomerang , the didgeridoo and Indigenous Australian music have become symbols of modern Australia . Governor Arthur Phillip hoists the British flag over
the new colony at Sydney in 1788 South Australian suffragette Catherine Helen Spence ( 1825 -- 1910 ) . The Australian colonies established democratic
parliaments from the 1850s and began to grant women the vote in the 1890s . The arrival of the first British settlers at what is now Sydney in 1788 introduced
Western civilisation to the Australian continent .
</item 10>
=== END ITEM id=0a347f33c64441f4 ===

=== ITEM id=47e858ac35dd35da ===
Gold sentences and facts:
Sentence 0: In most of the world speed limit signs display the limitation within a red circle . This design follows the style set out by the Vienna Convention on Road Signs and Signals , with the exception of United States and Canada . All countries globally worldwide ( except the United Kingdom and United States ) use metric speed limits in kilometres per hour . Metric signs are optional in the United States though less commonly seen , while Samoa uses both units . In Ontario , Canada , the type , location , and frequency of speed limit signs is covered by regulation 615 of the Ontario Highway Traffic Act .
  Fact 0.0: In most of the world, speed limit signs display the limitation within a red circle.
  Fact 0.1: The red-circle speed limit sign design follows the style set out by the Vienna Convention on Road Signs and Signals.
  Fact 0.2: The Vienna Convention style is not followed by the United States and Canada.
  Fact 0.3: All countries worldwide, except the United Kingdom and the United States, use metric speed limits in kilometres per hour.
  Fact 0.4: Metric speed limit signs are optional in the United States.
  Fact 0.5: Metric speed limit signs are less commonly seen in the United States.
  Fact 0.6: Samoa uses both units for speed limits.
  Fact 0.7: In Ontario, Canada, the type, location, and frequency of speed limit signs is covered by regulation 615 of the Ontario Highway Traffic Act.

Retrieved items:
<item 1>
Speed limit
The speeding fatality rate for local roads is three times that for Interstates Jump up ^ http://www.pe.com/articles/speed-776756-survey-limit.html Jump up ^
http://caselaw.findlaw.com/mt-supreme-court/1110919.html Jump up ^
http://vincenzeslaw.com/4-ways-virginia-reckless-driving-charges-handled-differently-fairfax-county-arlington-alexandria-stafford/ Jump up ^ `` Traffic
Operations : Road Signs '' . Government of South Australia . 2002 - 08 - 23 ( last update date ) . Retrieved 2010 - 09 - 26 . Most regulatory signs are
rectangular with a black legend on a white background ... ( exceptions include ) speed restriction signs ( symbol enclosed in a red circle ) '' Check date
values in : date = ( help ) Jump up ^ `` Speed Zoning Information '' ( PDF ) . Institute of Transportation Engineers . 22 March 2004 . Retrieved 2009 - 09 - 17
. Jump up ^ MUTCD Sections 2B.
</item 1>

<item 2>
Speed limit
In the European Union , large signposts showing the national ( default ) speed limits of the respective country are usually erected immediately after border
crossings , with a repeater sign some 200 to 500 m ( 660 to 1,640 ft ) after the first sign . Some places provide an additional `` speed zone ahead '' ahead of
the restriction and speed limit reminder signs may appear at regular intervals which may be painted on the road surface . Signs are normally placed on both
sides of the road and in some places there are small ( less than 1 / 4 the size of the sign ) rectangular orange reflector flags attached to both upper right
corners of both signs . The speed limit sign marking the new speed zone may also have the orange flags ; this practice can be observed in New York on highways
where the speed limit varies such as New York State Route 17 . Speed limits on European Motorways in km / h Signage in many countries , especially in Europe ,
conforms to the Vienna Convention on Road Signs and Signals using black text with a red circle on a white background . In the United States , the signs are
usually rectangular with the words `` SPEED LIMIT '' and the values in black on a white background .
</item 2>

<item 3>
Speed limit
Jump up ^ MUTCD Sections 2B. 13 - 16 # 12 Jump up ^ `` California Vehicle Code § 21400 ( b ) '' . The State of California . January 1 , 2012 . Retrieved 2013 -
07 - 28 . Jump up ^ Transportation Research Board of the National Academies ; Kay Fitzpatrick ; Paul Carlson ; Marcus A. Brewer ; Mark D. Wooldridge ; Shaw -
Pin Miaou ; Texas Transportation Institute ( October 6 , 2003 ) . NCHRP Report 504 -- Design Speed , Operating Speed , and Posted Speed Practices ( PDF ) .
onlinepubs.trb.org . Washington , D.C. : National Academies Press . p. 88 . ISBN 0 - 309 - 08767 - 8 . ISSN 0077 - 5614 . Archived from the original on April 12
, 2014 . Retrieved 2009 - 09 - 16 .
</item 3>

<item 4>
Speed limit
) SADC Road Traffic Signs Manual ( SADC ) Traffic Signs Regulations and General Directions ( UK ) Traffic law and safety Rules of the road All - way stop
Assured Clear Distance Ahead Australian Road Rules Boulevard rule Green Cross Code Move over law New Zealand Road Code Overtaking Right - and left - hand
traffic Right - of - way School bus traffic stop laws Traffic code Turn on red Vienna Convention on Road Traffic Road user guides Driver 's manual The Highway
Code The Highway Code ( Malta ) Road Users ' Code Rules of the Road ( Ireland ) Enforcement Breathalyzer Camera Enforcement Red light Highway patrol / State
police Parking enforcement Road traffic control Textalyzer Traffic court Traffic guard Traffic stop Traffic ticket Warning Speed limit Advisory speed limit
Assured Clear Distance Ahead National Maximum Speed Law School zone Speed limit enforcement Speed limits by country Category : Speed limits by country Moving
violations Driving under the influence Canada United States Distracted driving Mobile phones and driving Texting while driving Reckless driving Street racing
Tailgating Vehicular homicide Wrong - way driving Driver licensing Commercial driver 's license European driving licence Graduated driver licensing
Inter-American Driving Permit International Driving Permit Learner 's permit Point system Driving licences by country List of countries by minimum driving age
Traffic violations reciprocity Commercial Driver 's License
</item 4>

<item 5>
Speed limit
The U.S. Federal Highway Administration 's Manual on Uniform Traffic Control Devices provides guidelines for the appearance of speed limit signs . In Alaska and
California , speed limits are often labeled `` MAXIMUM SPEED '' instead . In Oregon , most speed limit signs just say `` SPEED '' and the number . Canada has
similar signs bearing the legend `` MAXIMUM '' and in km / h instead of mph . `` Maximum '' is used instead of `` Speed Limit '' because it has similar meanings
in English and French . Australian signs are rectangular but have a red circle like the Vienna Convention signs . The speed limit is commonly set at or below
the 85th percentile operating speed ( being the speed which no more than 15 % of traffic is exceeding ) and in the US is frequently set 4 to 8 mph ( 6 to 13 km
/ h ) below that speed . Thus , if the 85th percentile operating speed as measured by a Traffic and Engineering Survey exceeds the design speed , legal
protection is given to motorists traveling at such speeds ( design speed is `` based on conservative assumptions about driver , vehicle and roadway
characteristics '' ) .
</item 5>

<item 6>
Speed limit
Per distance traveled , consequences of inappropriate speed are more frequent on lower speed , lower quality roads ; in the United States , for example , the ``
speeding fatality rate for local roads is three times that for Interstates '' Citations for violations of the basic speed law without a crash have sometimes
been ruled unfairly vague or arbitrary , hence a violation the due process of law , at least in the State of Montana . Even within states , differing
jurisdictions ( counties and cities ) choose to prosecute similar cases with differing approaches . Maximum speed limits ( edit ) A speed limit sign in Ontario
In France city sign has value of speed limit A street limited at 30 km / h End of speed limit , previous speed limit applies Most public roads in most countries
have a legally assigned numerical maximum speed limit which applies on all roads unless otherwise stated ; lower speed limits are often shown on a sign at the
start of the restricted section , although the presence of streetlights or the physical arrangement of the road may sometimes also be used instead . A posted
speed limit may only apply to that road or to all roads beyond the sign that defines them depending on local laws .
</item 6>

<item 7>
Speed limit
In international European roads , speed should be taken into account at design stage . Maximum speed recommended by international European road design standard
Road classification 60 km / h 80 km / h ( 50 mph ) 100 km / h ( 60 mph ) 120 km / h ( 75 mph ) 140 km / h ( 85 mph ) Motorway x 80 100 120 140 Express road 60
80 100 120 x Road 60 80 100 x x Minimum speed limits ( edit ) UK minimum speed limit sign Some roads also have `` minimum speed limits '' , where slow speeds
can impede traffic flow or be dangerous . Signs often use blue circles based on the obligatory signs of the Vienna Convention on Road Signs and Signals . A
Japanese minimum speed sign has the same design as a maximum speed sign , but with a horizontal line below the number . In the United States of America , they
are also identical to their respective maximum speed limit signs with the exception of the text `` MINIMUM SPEED '' . This minimum speed is not so common , as
the risk usually raise with higher speed and not with lower speed .
</item 7>

<item 8>
Speed limit
Variable speed limits ( edit ) Example variable speed limit sign in the United States Digital speed limit sign for variable speed limits In 1965 , the first
known experiments with variable speed limit signs took place on a 30 km stretch of German motorway A8 between Munich and the border city of Salzburg , Austria .
Mechanically variable message signs could display speeds of 60 , 80 and 100 km / h , `` danger zone '' or `` accident '' . Personnel monitored traffic using
video technology , and manually controlled the signage . Beginning in the 1970s , additional advanced traffic control systems were put into service . Modern
motorway control systems can work without human intervention using various types of sensors to measure traffic flow and weather conditions . In 2009 , 1,300 km
( 810 mi ) of German motorways were equipped with such systems . In the late 1960s , heavily traveled portions of the New Jersey Turnpike began using variable
speed limit signs , in combination with variable message signs . Officials can adjust the speed limit according to weather , traffic conditions , and
construction .
</item 8>

<item 9>
Speed limit
p. 343 . External links ( edit ) Wikimedia Commons has media related to Speed limit signs and Driver speed and relative risk of an accident . Global map of
speed limits from Wikispeedia A Walk Through The History of Speed Limit Signs in The US John F. Carr 's State Traffic and Speed Laws Global map of speed limits
from OpenStreetMap Traffic signs Signs Warning sign Regulatory sign Prohibitory traffic sign Mandatory sign Special regulation sign Stop sign Yield sign
Crossbuck Dead End sign One - way traffic Speed limit ( by country ) Advisory speed limit Information sign Direction , position , or indication sign Variable -
message sign Bilingual sign Driver location signs Logo sign Priority to the right By comparison Comparison of European road signs Comparison of MUTCD -
influenced traffic signs Comparison of traffic signs in English - speaking countries By country Africa Botswana Mauritius South Africa Zimbabwe Asia Azerbaijan
Bangladesh Brunei Cambodia China Hong Kong India Indonesia Iran Israel Japan Laos Malaysia Nepal Pakistan Singapore South Korea Sri Lanka Thailand Turkey United
Arab Emirates Vietnam Europe Austria Bosnia and Herzegovina Czech Republic Denmark Estonia Finland France Germany Greece Iceland Ireland Italy Liechtenstein
Netherlands Norway Poland Sweden Switzerland Ukraine United Kingdom Scotland Wales North America Canada Mexico United States Oceania Australia New Zealand South
America Argentina Brazil Chile Colombia Lights Traffic
</item 9>

<item 10>
Speed limit
; for example California Vehicle Code section 22350 which states that `` No person shall drive a vehicle upon a highway at a speed greater than is reasonable
... and in no event at a speed which endangers the safety of persons or property '' . The basic speed law is almost always supplemented by specific maximum or
minimum limits , but applies regardless . The reasonable speed may be different than the posted speed limit in conditions such as fog , heavy rain , ice , snow
, gravel , sharp corners , blinding glare , darkness , crossing traffic , or when there is an obstructed view of orthogonal traffic -- such as by road curvature
, parked cars , vegetation , or snow banks -- thus limiting the Assured Clear Distance Ahead ( ACDA ) . Basic speed laws are statutized reinforcements of the
centuries - old common law negligence doctrine as specifically applied to vehicular speed .
</item 10>
=== END ITEM id=47e858ac35dd35da ===
