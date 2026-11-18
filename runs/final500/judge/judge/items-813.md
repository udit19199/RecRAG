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

=== ITEM id=b8a3d4aeaaac31dc ===
Gold sentences and facts:
Sentence 0: Callability -- Some bonds give the issuer the right to repay the bond before the maturity date on the call dates ; see call option . These bonds are referred to as callable bonds . Most callable bonds allow the issuer to repay the bond at par . With some bonds , the issuer has to pay a premium , the so - called call premium . This is mainly the case for high - yield bonds . These have very strict covenants , restricting the issuer in its operations . To be free from these covenants , the issuer can repay the bonds early , but only at a high cost .
  Fact 0.0: Some bonds give the issuer the right to repay the bond before the maturity date on the call dates.
  Fact 0.1: Bonds that give the issuer this right are referred to as callable bonds.
  Fact 0.2: Most callable bonds allow the issuer to repay the bond at par.
  Fact 0.3: With some bonds, the issuer has to pay a premium, called the call premium.
  Fact 0.4: Call premiums are mainly the case for high-yield bonds.
  Fact 0.5: High-yield bonds have very strict covenants that restrict the issuer in its operations.
  Fact 0.6: To be free from these covenants, the issuer can repay high-yield bonds early.
  Fact 0.7: Repaying high-yield bonds early to be free from covenants is possible only at a high cost.

Retrieved items:
<item 1>
Bond (finance)
Optionality : Occasionally a bond may contain an embedded option ; that is , it grants option - like features to the holder or the issuer : Callability -- Some
bonds give the issuer the right to repay the bond before the maturity date on the call dates ; see call option . These bonds are referred to as callable bonds .
Most callable bonds allow the issuer to repay the bond at par . With some bonds , the issuer has to pay a premium , the so - called call premium . This is
mainly the case for high - yield bonds . These have very strict covenants , restricting the issuer in its operations . To be free from these covenants , the
issuer can repay the bonds early , but only at a high cost . Putability -- Some bonds give the holder the right to force the issuer to repay the bond before the
maturity date on the put dates ; see put option . These are referred to as retractable or putable bonds . Call dates and put dates -- the dates on which
callable and putable bonds can be redeemed early . There are four main categories : A Bermudan callable has several call dates , usually coinciding with coupon
dates .
</item 1>

<item 2>
Bond (finance)
The name `` coupon '' arose because in the past , paper bond certificates were issued which had coupons attached to them , one for each interest payment . On
the due dates the bondholder would hand in the coupon to a bank in exchange for the interest payment . Interest can be paid at different frequencies : generally
semi-annual , i.e. every 6 months , or annual . Bond issued by the Dutch East India Company in 1623 Yield ( edit ) The yield is the rate of return received from
investing in the bond . It usually refers either to The current yield , or running yield , which is simply the annual interest payment divided by the current
market price of the bond ( often the clean price ) . The yield to maturity , or redemption yield , which is a more useful measure of the return of the bond .
This takes into account the current market price , and the amount and timing of all remaining coupon payments and of the repayment due on maturity . It is
equivalent to the internal rate of return of a bond . Credit quality ( edit ) The quality of the issue refers to the probability that the bondholders will
receive the amounts promised at the due dates .
</item 2>

<item 3>
Bond (finance)
Book - entry bond is a bond that does not have a paper certificate . As physically processing paper bonds and interest coupons became more expensive , issuers (
and banks that used to collect coupon interest for depositors ) have tried to discourage their use . Some book - entry bond issues do not offer the option of a
paper certificate , even to investors who prefer them . Lottery bonds are issued by European and other states . Interest is paid as on a traditional fixed rate
bond , but the issuer will redeem randomly selected individual bonds within the issue according to a schedule . Some of these redemptions will be for a higher
value than the face value of the bond . War bond is a bond issued by a government to fund military operations during wartime . This type of bond has low return
rate . Serial bond is a bond that matures in installments over a period of time . In effect , a $100,000 , 5 - year serial bond would mature in a $20,000
annuity over a 5 - year interval . Revenue bond is a special type of municipal bond distinguished by its guarantee of repayment solely from revenues generated
by a specified revenue - generating entity associated with the purpose of the bonds .
</item 3>

<item 4>
Bond (finance)
The length of time until the maturity date is often referred to as the term or tenure or maturity of a bond . The maturity can be any length of time , although
debt securities with a term of less than one year are generally designated money market instruments rather than bonds . Most bonds have a term of up to 30 years
. Some bonds have been issued with terms of 50 years or more , and historically there have been some issues with no maturity date ( irredeemable ) . In the
market for United States Treasury securities , there are three categories of bond maturities : short term ( bills ) : maturities between one and five years ;
medium term ( notes ) : maturities between six and twelve years ; long term ( bonds ) : maturities longer than twelve years . Coupon ( edit ) The coupon is the
interest rate that the issuer pays to the holder . Usually this rate is fixed throughout the life of the bond . It can also vary with a money market index ,
such as LIBOR , or it can be even more exotic .
</item 4>

<item 5>
Bond (finance)
The most common types of bonds include municipal bonds and corporate bonds . The bond is a debt security , under which the issuer owes the holders a debt and (
depending on the terms of the bond ) is obliged to pay them interest ( the coupon ) or to repay the principal at a later date , termed the maturity date .
Interest is usually payable at fixed intervals ( semiannual , annual , sometimes monthly ) . Very often the bond is negotiable , that is , the ownership of the
instrument can be transferred in the secondary market . This means that once the transfer agents at the bank medallion stamp the bond , it is highly liquid on
the secondary market . Thus a bond is a form of loan or IOU : the holder of the bond is the lender ( creditor ) , the issuer of the bond is the borrower (
debtor ) , and the coupon is the interest . Bonds provide the borrower with external funds to finance long - term investments , or , in the case of government
bonds , to finance current expenditure . Certificates of deposit ( CDs ) or short - term commercial paper are considered to be money market instruments and not
bonds : the main difference is the length of the term of the instrument .
</item 5>

<item 6>
Bond (finance)
For other uses , see Bond ( disambiguation ) . Securities Securities ( show ) Banknote Bond Debenture Derivative Stock Markets ( show ) Stock market Commodity
market Foreign exchange market Futures exchange Over-the - counter market ( OTC ) Spot market Bonds by coupon ( show ) Fixed rate bond Floating rate note
Inflation - indexed bond Perpetual bond Zero - coupon bond Commercial paper Bonds by issuer ( show ) Corporate bond Government bond Municipal bond Pfandbrief
Equities ( stocks ) ( show ) Stock Share Initial public offering ( IPO ) Short selling Investment funds ( show ) Mutual fund Closed - end fund Exchange - traded
fund ( ETF ) Hedge fund Index fund Segregated fund Structured finance ( show ) Securitization Agency security Asset - backed security Mortgage - backed security
Commercial mortgage - backed security Residential mortgage - backed security Tranche Collateralized debt obligation Collateralized fund obligation
Collateralized mortgage obligation Credit - linked note Unsecured debt Derivatives ( show ) Forward Futures Option Swap Warrant Credit derivative Hybrid
security In finance , a bond is an instrument of indebtedness of the bond issuer to the holders . The most common types of bonds include municipal bonds and
corporate bonds .
</item 6>

<item 7>
Bond (finance)
The latter are often issued in tranches . The senior tranches get paid back first , the subordinated tranches later . Covered bonds are backed by cash flows
from mortgages or public sector assets . Contrary to asset - backed securities the assets for such bonds remain on the issuers balance sheet . Perpetual bonds
are also often called perpetuities or ' Perps ' . They have no maturity date . The most famous of these are the UK Consols , which are also known as Treasury
Annuities or Undated Treasuries . Some of these were issued back in 1888 and still trade today , although the amounts are now insignificant . Some ultra-long -
term bonds ( sometimes a bond can last centuries : West Shore Railroad issued a bond which matures in 2361 ( i.e. 24th century ) are virtually perpetuities from
a financial point of view , with the current value of principal near zero . Bearer bond is an official certificate issued without a named holder . In other
words , the person who has the paper certificate can claim the value of the bond .
</item 7>

<item 8>
Bond (finance)
Bonds and stocks are both securities , but the major difference between the two is that ( capital ) stockholders have an equity stake in a company ( that is ,
they are owners ) , whereas bondholders have a creditor stake in the company ( that is , they are lenders ) . Being a creditor , bondholders have priority over
stockholders . This means they will be repaid in advance of stockholders , but will rank behind secured creditors , in the event of bankruptcy . Another
difference is that bonds usually have a defined term , or maturity , after which the bond is redeemed , whereas stocks typically remain outstanding indefinitely
. An exception is an irredeemable bond , such as a consol , which is a perpetuity , that is , a bond with no maturity .
</item 8>

<item 9>
Bond (finance)
A European callable has only one call date . This is a special case of a Bermudan callable . An American callable can be called at any time until the maturity
date . A death put is an optional redemption feature on a debt instrument allowing the beneficiary of the estate of a deceased bondholder to put ( sell ) the
bond back to the issuer at face value in the event of the bondholder 's death or legal incapacitation . This is also known as a `` survivor 's option '' .
Sinking fund provision of the corporate bond indenture requires a certain portion of the issue to be retired periodically . The entire bond issue can be
liquidated by the maturity date ; if not , the remainder is called balloon maturity . Issuers may either pay to trustees , which in turn call randomly selected
bonds in the issue , or , alternatively , purchase bonds in open market , then return them to trustees . Bonds are often identified by its international
securities identification number , or ISIN , which is a 12 digit alphanumeric code that uniquely identifies debt securities .
</item 9>

<item 10>
Bond (finance)
The net proceeds that the issuer receives are thus the issue price , less issuance fees . The market price of the bond will vary over its life : it may trade at
a premium ( above par , usually because market interest rates have fallen since issue ) , or at a discount ( price below par , if market rates have risen or
there is a high probability of default on the bond ) . Others ( edit ) Indentures and Covenants -- An indenture is a formal debt agreement that establishes the
terms of a bond issue , while covenants are the clauses of such an agreement . Covenants specify the rights of bondholders and the duties of issuers , such as
actions that the issuer is obligated to perform or is prohibited from performing . In the U.S. , federal and state securities and commercial laws apply to the
enforcement of these agreements , which are construed by courts as contracts between issuers and bondholders . The terms may be changed only with great
difficulty while the bonds are outstanding , with amendments to the governing document generally requiring approval by a majority ( or super-majority ) vote of
the bondholders .
</item 10>
=== END ITEM id=b8a3d4aeaaac31dc ===

=== ITEM id=b7dfa4a3d539edb4 ===
Gold sentences and facts:
Sentence 0: No . Title Original air date `` The Dark Dream Begins '' `` Kuroki Yume no Hajimari '' ( 黒き 夢 の はじまり ) August 28 , 2016 Elizabeth Liones goes to the Boar Hat , where all her friends are still asleep following the festival . After Elizabeth is greeted by Hawk outside , Meliodas introduces her to a meat pie made with vegetables that looks like Hawk , but it turns out to be unfit for human consumption . Meliodas has the idea of using pork as the main ingredient , which causes Hawk to make haste out of the bar . Meliodas comes up with a contest for his friends , in which he will obey any order for a day from whoever catches Hawk first . However , Hawk manages to outsmart them due to his small size . Hawk returns to the Boar Hat and eats food scraps made by Ban , who is declared the winner as a result . At night , Meliodas takes Elizabeth back to the palace while also assuring that his purpose in life is to protect her . Meliodas then goes to the catacombs to see the Horn of Cernunnos . When the goddess vessel taunts him for his reputation as a demon , Meliodas destroys the Horn of Cernunnos in retaliation . `` Our Fighting Festival '' `` Futari no Ken'ka Matsuri '' ( 二 人 の 喧嘩 祭り ) September 4 , 2016 Ban walks up to a mountaintop to finally settle matters with Meliodas , but the former is surprised to see Elizabeth and Hawk as well as the other members of the Seven Deadly Sins and the Holy Knights as part of an audience . Merlin casts a gigantic cubic barrier around Meliodas and Ban in order for the spectators to safely watch the fight . Meanwhile , King and Hauser have their own fight to win the attention of Diane . Ban initiates a brutal attack on Meliodas that somehow weakens most of the spectators . The fight ends in a stalemate after the two of them exchange rapid punches that cause the barrier to break . They remain friends and head back down to the Boar Hat for a drink . `` In Pursuit of First Love '' `` Hatsukoi o Oikakete '' ( 初恋 を 追いかけ て ) September 11 , 2016 King still hesitates to confess his feelings for Diane . Meanwhile , Diane leaves the Boar Hat and heads to a construction site in town , recalling the devastating collateral damage she caused due to her immense size . The construction workers there apologize to her and thank her for saving the kingdom . When King comes to see Diane , the two decide to play a game of tag across town . Outside a church , a wedding ceremony is interrupted by a demon , in which Diane and King combine their skills in defeating the demon . King asks Diane if she regained her memories , and she answers him with a kiss on his cheek . The moment is ruined when a brick suddenly falls on his head , and he later wakes up next to her , having forgotten what they did during the day . `` The Shape of Love '' `` Ai no Katachi '' ( 愛 の かたち ) September 18 , 2016 Merlin agrees to be a waitress at the Boar Hat , but first she wants Meliodas and Gowther to follow Gilthunder to find out if he is being stalked by someone . Meanwhile , Gilthunder , Hauser and Griamore are each hit with falling debris in the streets , unaware of the situation . Gowther is distracted by a boy named Pelliot , who asks him to retrieve a fat cat off a roof . Since Gowther reminds Pelliot of his mother , who recently passed away , Gowther transforms into Pelliot 's mother for a moment and disappears when Pelliot 's father finds Pelliot . Meliodas finds Gilthunder , Hauser and Griamore at a restaurant , where he purposely spills coffee to expose Vivian , revealed as the stalker . Merlin punishes Vivian by planting a cursed ring on her finger , which will bring pain to her each time the curse is activated , and Merlin explains that the Ten Commandments are the highest class in the demon race . At the Boar Hat , Merlin quits her job as a waitress , while Hawk and Gowther fill in for the position , much to the guys ' dismay .
  Fact 0.0: The episode 'The Dark Dream Begins' (Japanese title 'Kuroki Yume no Hajimari', 黒き 夢 の はじまり) first aired on August 28, 2016.
  Fact 0.1: Elizabeth Liones goes to the Boar Hat, where all her friends are still asleep following the festival.
  Fact 0.2: After Elizabeth is greeted by Hawk outside, Meliodas introduces her to a meat pie made with vegetables that looks like Hawk.
  Fact 0.3: The meat pie made with vegetables turns out to be unfit for human consumption.
  Fact 0.4: Meliodas has the idea of using pork as the main ingredient.
  Fact 0.5: Using pork as the main ingredient causes Hawk to make haste out of the bar.
  Fact 0.6: Meliodas comes up with a contest for his friends.
  Fact 0.7: In the contest, Meliodas will obey any order for a day from whoever catches Hawk first.
  Fact 0.8: Hawk manages to outsmart the contestants due to his small size.
  Fact 0.9: Hawk returns to the Boar Hat and eats food scraps made by Ban.
  Fact 0.10: Ban is declared the winner as a result.
  Fact 0.11: At night, Meliodas takes Elizabeth back to the palace.
  Fact 0.12: Meliodas assures Elizabeth that his purpose in life is to protect her.
  Fact 0.13: Meliodas then goes to the catacombs to see the Horn of Cernunnos.
  Fact 0.14: When the goddess vessel taunts Meliodas for his reputation as a demon, Meliodas destroys the Horn of Cernunnos in retaliation.
  Fact 0.15: The episode 'Our Fighting Festival' (Japanese title 'Futari no Ken'ka Matsuri', 二 人 の 喧嘩 祭り) first aired on September 4, 2016.
  Fact 0.16: Ban walks up to a mountaintop to finally settle matters with Meliodas.
  Fact 0.17: Ban is surprised to see Elizabeth and Hawk as well as the other members of the Seven Deadly Sins and the Holy Knights as part of an audience.
  Fact 0.18: Merlin casts a gigantic cubic barrier around Meliodas and Ban so the spectators can safely watch the fight.
  Fact 0.19: King and Hauser have their own fight to win the attention of Diane, meanwhile.
  Fact 0.20: Ban initiates a brutal attack on Meliodas that somehow weakens most of the spectators.
  Fact 0.21: The fight ends in a stalemate after the two of them exchange rapid punches that cause the barrier to break.
  Fact 0.22: Meliodas and Ban remain friends and head back down to the Boar Hat for a drink.
  Fact 0.23: The episode 'In Pursuit of First Love' (Japanese title 'Hatsukoi o Oikakete', 初恋 を 追いかけ て) first aired on September 11, 2016.
  Fact 0.24: King still hesitates to confess his feelings for Diane.
  Fact 0.25: Diane leaves the Boar Hat and heads to a construction site in town.
  Fact 0.26: Diane recalls the devastating collateral damage she caused due to her immense size.
  Fact 0.27: The construction workers at the site apologize to Diane and thank her for saving the kingdom.
  Fact 0.28: When King comes to see Diane, the two decide to play a game of tag across town.
  Fact 0.29: Outside a church, a wedding ceremony is interrupted by a demon.
  Fact 0.30: Diane and King combine their skills in defeating the demon.
  Fact 0.31: King asks Diane if she regained her memories, and she answers him with a kiss on his cheek.
  Fact 0.32: The moment is ruined when a brick suddenly falls on King's head.
  Fact 0.33: King later wakes up next to Diane, having forgotten what they did during the day.
  Fact 0.34: The episode 'The Shape of Love' (Japanese title 'Ai no Katachi', 愛 の かたち) first aired on September 18, 2016.
  Fact 0.35: Merlin agrees to be a waitress at the Boar Hat.
  Fact 0.36: Before taking the waitress job, Merlin wants Meliodas and Gowther to follow Gilthunder to find out if he is being stalked by someone.
  Fact 0.37: Gilthunder, Hauser and Griamore are each hit with falling debris in the streets, unaware of the situation.
  Fact 0.38: Gowther is distracted by a boy named Pelliot, who asks him to retrieve a fat cat off a roof.
  Fact 0.39: Gowther reminds Pelliot of his mother, who recently passed away.
  Fact 0.40: Gowther transforms into Pelliot's mother for a moment.
  Fact 0.41: Gowther disappears when Pelliot's father finds Pelliot.
  Fact 0.42: Meliodas finds Gilthunder, Hauser and Griamore at a restaurant.
  Fact 0.43: Meliodas purposely spills coffee to expose Vivian.
  Fact 0.44: Vivian is revealed as the stalker.
  Fact 0.45: Merlin punishes Vivian by planting a cursed ring on her finger.
  Fact 0.46: The cursed ring will bring pain to Vivian each time the curse is activated.
  Fact 0.47: Merlin explains that the Ten Commandments are the highest class in the demon race.
  Fact 0.48: At the Boar Hat, Merlin quits her job as a waitress.
  Fact 0.49: Hawk and Gowther fill in for the waitress position.
  Fact 0.50: Hawk and Gowther filling in for the waitress position is much to the guys' dismay.

Retrieved items:
<item 1>
List of The Seven Deadly Sins episodes
List of the Seven Deadly Sins episodes - wikipedia List of the Seven Deadly Sins episodes Jump to : navigation , search The Seven Deadly Sins is a Japanese
manga series written and illustrated by Nakaba Suzuki . The series debuted on MBS and other Japan News Network stations on October 5 , 2014 . The series follows
Elizabeth , the third princess of the Kingdom of Liones , who is in search of the Seven Deadly Sins , a group of former Holy Knights who were disbanded after
the kingdom was overthrown . She enlists the help of Meliodas and Hawk , the captain of the Seven Deadly Sins and his talking pet pig , to assemble the
remaining members of the Seven Deadly Sins and take back the kingdom from the Holy Knights .
</item 1>

<item 2>
List of The Seven Deadly Sins episodes
A second OVA composed of nine humorous shorts was shipped with the limited edition of the sixteenth volume of the manga , released on August 12 , 2015 . The
first The Seven Deadly Sins anime series was licensed for English release by Netflix as its second exclusive anime , following their acquisition of Knights of
Sidonia . All 24 episodes were released on November 1 , 2015 in both subtitled or English dub formats . The Seven Deadly Sins Revival of The Commandments first
opening theme song titled Howling is a collaboration between Flow and Granrodeo and first ending theme song is Beautiful performed by Anly . Contents ( hide ) 1
Episode list 1.1 The Seven Deadly Sins 1.2 The Seven Deadly Sins : Signs of Holy War 1.3 The Seven Deadly Sins : Revival of The Commandments 2 OVAs 3 References
Episode list ( edit ) The Seven Deadly Sins ( edit ) No .
</item 2>

<item 3>
List of The Seven Deadly Sins episodes
14 `` A Reader of Books '' `` Hon o Yomu Hito '' ( 本 を 読む ひと ) January 18 , 2015 News about several sightings of an Armor Giant somewhere in Ordan Forest
reaches Helbram , who quickly sends Dawn Roar , an elite group of Holy Knights , to search for the Armor Giant . Meanwhile , Guila visits her younger brother
Zeal , and she later kills three thugs who bullied him for his spare change . Meliodas and his friends split up in the nearby village . Elizabeth and Hawk
encounter a boy named Alan , who is looking for some glue to repair armor . Upon hearing the low growls that echo in the forest , Meliodas , Ban and King find
the Armor Giant being surrounded by Dawn Roar , but Meliodas , Ban and King are no match against Dawn Roar . Just as it seems that the Armor Giant is about to
be slain , Alan intercepts and reveals himself as Gowther , the Goat 's Sin of Lust .
</item 3>

<item 4>
List of The Seven Deadly Sins episodes
Things return to normal inside the kingdom , and a festival is held at night . The next day , Elizabeth joins Meliodas , Hawk , Diane , Ban , King , Gowther and
Merlin in finding the last member of the Seven Deadly Sins , Escanor , the Lion 's Sin of Pride . The Seven Deadly Sins : signs of Holy war ( edit ) No . Title
Original air date `` The Dark Dream Begins '' `` Kuroki Yume no Hajimari '' ( 黒き 夢 の はじまり ) August 28 , 2016 Elizabeth Liones goes to the Boar Hat , where all
her friends are still asleep following the festival . After Elizabeth is greeted by Hawk outside , Meliodas introduces her to a meat pie made with vegetables
that looks like Hawk , but it turns out to be unfit for human consumption . Meliodas has the idea of using pork as the main ingredient , which causes Hawk to
make haste out of the bar . Meliodas comes up with a contest for his friends , in which he will obey any order for a day from whoever catches Hawk first .
</item 4>

<item 5>
List of The Seven Deadly Sins episodes
`` Revival of the Demon Clan '' `` Majin - zoku Fukkatsu '' ( 魔神 族 復活 ) January 13 , 2018 After the defeat of demonized Hendrickson , The Seven Deadly Sins are
commended for their bravery and courage by King Bartra . However , just before the ceremony , Ban and King left the group . Earlier , Ban tells Meliodas that he
will leave the group . As Ban leaves the Boar Hat Bar , King sees him and asks where he is going . Ban tells King that he wants to go to the Fairy King Forest .
King then accompanies Ban on their journey . Meanwhile , it reveals that Hendricksen has survived that assault from the Sins and he is shocked upon seeing
Dreyfus alive . Dreyfus then summons the Commandments , ten demonized warriors with the intent of destroying the Sins .
</item 5>

<item 6>
List of The Seven Deadly Sins episodes
15 `` Unholy Knight '' `` Anhōryi Naito '' ( アンホーリィ ・ ナイト ) January 25 , 2015 Gowther gives the head of the Armor Giant to the members of Dawn Roar in order to
convince them to withdraw from battle . However , the demon breaks through the armor and attacks Meliodas , Ban , King and Gowther . Meliodas lowers his guard
upon noticing a Holy Knight merged into the demon 's chest , but Gowther uses his Sacred Treasure , the Twin Bow Herritt , to stop the demon . Meanwhile , Cain
visits Elizabeth in the Boar Hat , where he tells her the story of Liz , a knight from an enemy kingdom who became Meliodas 's past lover after he saved her
from being sentenced to death . Ban gouges out the heart of the Holy Knight , recognized as Guila 's father Dale , yet the demon still remains alive . Elizabeth
rushes to Meliodas and gives him Liz 's sword left behind , which allows him to destroy the demon . It is realized that Dale was one of Hendrickson 's failed
demon experiments .
</item 6>

<item 7>
List of The Seven Deadly Sins episodes
It is realized that Dale was one of Hendrickson 's failed demon experiments . 16 `` The Legends , Provoked '' `` Karitaterareru Densetsu - tachi '' ( 駆り立て られる
伝説 たち ) February 1 , 2015 Gowther is welcomed by the others into their group , but he soon finds out the true secrets behind them by delving deep into their
memories and emotions . Meanwhile , Hendrickson and Helbram have the hilt of Meliodas 's sword in possession , in which they plan to use it as a fragment of a
ritual relic , known as the Coffin of Eternal Darkness , to resurrect the red demon race as a means to start a holy war . At the Boar Hat , Holy Knight Vivian
suddenly appears and kidnaps Elizabeth and Hawk . Diane throws Meliodas , Ban and Gowther toward the Kingdom of Liones , where the three are bombarded by an
army of Holy Knights on the north side of the castle .
</item 7>

<item 8>
List of The Seven Deadly Sins episodes
Title Original air date `` The Seven Deadly Sins '' `` Nanatsu no Taizai '' ( 七 つの 大罪 ) October 5 , 2014 Elizabeth Liones , the third princess of the Kingdom of
Liones , is on a mission to find the Seven Deadly Sins , a group of extremely powerful but dangerous group of Holy Knights who are now wanted criminals for
supposedly betraying the kingdom . She finds Meliodas , the owner of a bar named the Boar Hat , accompanied by his pet companion Hawk , a talking pig . When
attacked by apprentice Holy Knight Twigo , Meliodas quickly overcomes the opponent , but reveals his identity in the process as the Dragon 's Sin of Wrath and
the captain of the Seven Deadly Sins . Meliodas and Hawk join Elizabeth on her journey to find the other Deadly Sins in order to defeat the Holy Knights , who
have orchestrated a coup d'état to overthrow the king .
</item 8>

<item 9>
List of The Seven Deadly Sins episodes
Vivian tries to trap Meliodas , Gilthunder , Margaret and Arthur in illusionary places , but Merlin , the Boar 's Sin of Gluttony , makes her first appearance
and breaks the spell . 21 `` The Looming Threat '' `` Ima , Soko ni Semaru Kyōi '' ( 今 、 そこ に せまる 脅威 ) March 8 , 2015 Hawk finds Ban in the catacombs underneath
the kingdom , where Ban makes contact with a goddess vessel through the Horn of Cernunnos . Ban is granted the request of resurrecting Elaine , but only with
the condition that he kills Meliodas . Meanwhile , Meliodas , Merlin , Gilthunder , Margaret and Arthur enter the chamber where Elizabeth and Bartra are being
held captive . Merlin takes Arthur and Bartra to Camelot to safety , but Hendrickson soon causes an explosion in the chamber , showing himself embedded in
darkness and causing terror in the town with the experimented Holy Knights turned into demons .
</item 9>

<item 10>
List of The Seven Deadly Sins episodes
`` The Sword of the Holy Knight '' `` Seikishi no Ken '' ( 聖 騎士 の 剣 ) October 12 , 2014 Holy Knight Gilthunder plants his sword in the center of a town named
the Bernia Village , thus preventing the town from making the Bernia Ale , the primary source of income . After innumerable unsuccessful attempts by the
villagers to remove the sword , which blocked the water source , Meliodas easily removes the object . A celebration is held at the Boar Hat , and Meliodas is
hailed as a savior of the Bernia Village . Gilthunder , wanting to confirm whether the removal of his sword was an accident or not , throws a spear imbued with
his power towards the town . However , Meliodas easily stops and throws the spear back , destroying Fort Solgales . Meliodas , Elizabeth and Hawk decide to
proceed to the Forest of White Dreams .
</item 10>
=== END ITEM id=b7dfa4a3d539edb4 ===

=== ITEM id=e7b8e744a665288e ===
Gold sentences and facts:
Sentence 0: The American entry into World War I came in April 1917 , after more than two and a half years of efforts by President Woodrow Wilson to keep the United States out of the war . Apart from an Anglophile element urging early support for the British , American public opinion reflected that of the president : the sentiment for neutrality was particularly strong among Irish Americans , German Americans and Scandinavian Americans , as well as among church leaders and among women in general . On the other hand , even before World War I had broken out , American opinion had been more negative toward Germany than towards any other country in Europe . Over time , especially after reports of atrocities in Belgium in 1914 and following the sinking of the passenger liner RMS Lusitania in 1915 , the American people increasingly came to see Germany as the aggressor in Europe .
  Fact 0.0: The American entry into World War I came in April 1917.
  Fact 0.1: The American entry into World War I came after more than two and a half years of efforts by President Woodrow Wilson to keep the United States out of the war.
  Fact 0.2: Apart from an Anglophile element urging early support for the British, American public opinion reflected that of the president.
  Fact 0.3: The sentiment for neutrality was particularly strong among Irish Americans.
  Fact 0.4: The sentiment for neutrality was particularly strong among German Americans.
  Fact 0.5: The sentiment for neutrality was particularly strong among Scandinavian Americans.
  Fact 0.6: The sentiment for neutrality was particularly strong among church leaders.
  Fact 0.7: The sentiment for neutrality was particularly strong among women in general.
  Fact 0.8: Before World War I had broken out, American opinion had been more negative toward Germany than toward any other country in Europe.
  Fact 0.9: Over time, the American people increasingly came to see Germany as the aggressor in Europe.
  Fact 0.10: The shift toward seeing Germany as the aggressor was especially driven by reports of atrocities in Belgium in 1914.
  Fact 0.11: The shift toward seeing Germany as the aggressor was especially driven by the sinking of the passenger liner RMS Lusitania in 1915.

Retrieved items:
<item 1>
American entry into World War I
After February 1 , the communique noted , German U-boats would sink without warning belligerent and neutral ships found in a designated zone comprising waters
around Great Britain , France , and Italy , and in the eastern Mediterranean . The Admiralty made one minor exception : it would permit one American steamer a
week to sail between New York and Falmouth ( ... ) . Initially Germany would grant a period of grace , during which its submarines would not harm neutral ships
that either were en route to the war zone or had already arrived . Jump up ^ Compare : Doenecke , Justus D. ( 2011 ) . Nothing Less Than War : A New History of
America 's Entry Into World War I. Studies in Conflict , Diplomacy and Peace Series . University Press of Kentucky . p. 286 . ISBN 978 - 0 - 8131 - 3002 - 6 .
Retrieved 2015 - 12 - 07 . Several factors led to Wilson 's choice . Germany 's U-boat warfare was paramount .
</item 1>

<item 2>
American entry into World War I
Sinking of American merchant ships ( edit ) In early 1917 Berlin forced the issue . Its declared decision on 31 January 1917 to target neutral shipping in a
designated war - zone became the immediate cause of the entry of the United States into the war . Five American merchant ships went down in March . Outraged
public opinion now overwhelmingly supported Wilson when he asked Congress for a declaration of war on April 2 , 1917 . Public opinion , moralism , and national
interest ( edit ) Historians such as Ernest R. May have approached the process of American entry into the war as a study in how public opinion changed radically
in three years ' time . In 1914 most Americans called for neutrality , seeing the war a dreadful mistake and were determined to stay out . By 1917 the same
public felt just as strongly that going to war was both necessary and wise . Military leaders had little to say during this debate , and military considerations
were seldom raised . The decisive questions dealt with morality and visions of the future . The prevailing attitude was that America possessed a superior moral
position as the only great nation devoted to the principles of freedom and democracy .
</item 2>

<item 3>
American entry into World War I
American entry into World war I - wikipedia American entry into World war I Jump to : navigation , search The American entry into World War I came in April 1917
, after more than two and a half years of efforts by President Woodrow Wilson to keep the United States out of the war . Apart from an Anglophile element urging
early support for the British , American public opinion reflected that of the president : the sentiment for neutrality was particularly strong among Irish
Americans , German Americans and Scandinavian Americans , as well as among church leaders and among women in general . On the other hand , even before World War
I had broken out , American opinion had been more negative toward Germany than towards any other country in Europe . Over time , especially after reports of
atrocities in Belgium in 1914 and following the sinking of the passenger liner RMS Lusitania in 1915 , the American people increasingly came to see Germany as
the aggressor in Europe . As U.S.
</item 3>

<item 4>
American entry into World War I
International Encyclopedia of the First World War . NY Times main headline , April 2 , 1917 , President Calls for War Declaration , Stronger Navy , New Army of
500,000 Men , Full Cooperation With Germany 's Foes President Wilson 's War Address Map of Europe at the time of the US declaration of war on Germany at
omniatlas.com World War I : Declarations of War from Around the Globe - How America Entered the Great War Today in History : U.S.
</item 4>

<item 5>
American entry into World War I
The solution was a third route , a `` peace without victory '' , according to Wilson . On April 6 , 1917 , Congress declared war . In the Senate , the
resolution passed 82 to 6 , with Senators Harry Lane , William J. Stone , James Vardaman , Asle Gronna , Robert M. La Follette , Sr. , and George W. Norris
voting against it . In the House , the declaration passed 373 to 50 , with Claude Kitchin , a senior Democrat , notably opposing it . Another opponent was
Jeannette Rankin , who alone voted against entry into both World War I and World War II . Nearly all of the opposition came from the West and the Midwest .
Austria - Hungary ( edit ) The United States Senate , in a 74 to 0 vote , declared war on Austria - Hungary on December 7 , 1917 , citing Austria - Hungary 's
severing of diplomatic relations with the United States , its use of unrestricted submarine warfare and its alliance with Germany . The declaration passed in
the United States House of Representatives by a vote of 365 to 1 .
</item 5>

<item 6>
American entry into World War I
252 -- 282 . ^ Jump up to : H.J. Res. 169 : Declaration of War with Austria - Hungary , WWI , United States Senate ^ Jump up to : Jennifer K. Elsea ; Matthew C.
Weed ( April 18 , 2014 ) . `` Declarations of War and Authorizations for the Use of Military Force : Historical Background and Legal Implications '' ( PDF ) .
p. 9 . Archived from the original ( PDF ) on August 10 , 2006 . Retrieved July 21 , 2014 . Jump up ^ Eric W. Osborne , Britain 's economic blockade of Germany ,
1914 -- 1919 ( 2004 ) Jump up ^ Ernest May , The World War and American Isolation , 1914 -- 1917 ( 1959 ) p. 115 quote from Dec 1914 . Jump up ^ C.R.M.F.
Cruttwell , A history of the Great War , 1914 -- 1918 ( 2007 ) p. 191 Jump up ^ Edward House , The intimate papers of Colonel House : Vol 2 ( 1928 ) p.
</item 6>

<item 7>
American entry into World War I
Germany 's U-boat warfare was paramount . Had it not been for Berlin 's announcement of January 31 , the president probably would not have issued a call to arms
. Jump up ^ Justus D. Doenecke , Nothing Less Than War : A New History of America 's Entry into World War I ( 2011 ) ch 10 ^ Jump up to : May , Ernest R. ( 1966
) . The World War and American isolation , 1914 -- 1917 . Jump up ^ Knock , Thomas J. ( 1995 ) . To End All Wars : Woodrow Wilson and the Quest for a New World
Order . Jump up ^ Syrett , Harold C. ( 1945 ) . `` The Business Press and American Neutrality , 1914 -- 1917 '' . Mississippi Valley Historical Review . 32 ( 2
) : 215 -- 230 . doi : 10.2307 / 1898209 . JSTOR 1898209 .
</item 7>

<item 8>
American entry into World War I
As U.S. president , it was Wilson who made the key policy decisions over foreign affairs : while the country was at peace , the domestic economy ran on a
laissez - faire basis , with American banks making huge loans to Britain and France -- funds that were in large part used to buy munitions , raw materials , and
food from across the Atlantic . Until 1917 , Wilson made minimal preparations for a land war and kept the United States Army on a small peacetime footing ,
despite increasing demands for enhanced preparedness . He did however expand the United States Navy . In 1917 , with Russia experiencing political upheaval
following widespread disillusionment there over the war , and with Britain and France low on credit , Germany appeared to have the upper hand in Europe , while
Germany 's ally , the Ottoman Empire , clung stubbornly to its possessions in the Middle East . In the same year , Germany decided to resume unrestricted
submarine warfare against any vessel approaching British waters ; this attempt to starve Britain into surrender was balanced against the knowledge that it would
almost certainly bring the United States into the war .
</item 8>

<item 9>
American entry into World War I
See also ( edit ) World War I portal Causes of World War I Diplomatic history of World War I United States in World War I United States home front during World
War I Footnotes ( edit ) Jump up ^ Jeanette Keith ( 2004 ) . Rich Man 's War , Poor Man 's Fight : Race , Class , and Power in the Rural South during the First
World War . U. of North Carolina Press . pp. 1 -- 5 . ISBN 978 - 0 - 8078 - 7589 - 6 . Jump up ^ Barnes , Harry Elmer . The Genesis of the World War ( 1925 )
pp. 590 -- 591 Jump up ^ `` World War One '' . BBC History . Jump up ^ Link , Arthur S. ( 1972 ) . Woodrow Wilson and the Progressive Era , 1910 -- 1917 . New
York : Harper & Row . pp. 252 -- 282 . ^ Jump up to : H.J. Res.
</item 9>

<item 10>
American entry into World War I
1971 . Safford , Jeffrey J. Wilsonian Maritime Diplomacy , 1913 -- 1921 . 1978 . Smith , Daniel . The Great Departure : The United States and World War I , 1914
-- 1920 . 1965 . Sterba , Christopher M. Good Americans : Italian and Jewish Immigrants during the First World War . 2003 . 288 pp. online edition ISBN 0 - 19 -
514754 - 5 OCLC 49576532 Tuchman , Barbara W. ( 1985 ) . The Zimmermann Telegram . ISBN 0 - 345 - 32425 - 0 . Tucker , Robert W. Woodrow Wilson and the Great
War : Reconsidering America 's Neutrality , 1914 -- 1917 . 2007 . ISBN 978 - 0 - 8139 - 2629 - 2 Unger , Nancy C. Fighting Bob La Follette : The Righteous
Reformer Chapel Hill : University of North Carolina Press , 2000 .
</item 10>
=== END ITEM id=e7b8e744a665288e ===

=== ITEM id=ca2e7541a8e0f0cc ===
Gold sentences and facts:
Sentence 0: Costs are associated with particular goods using one of the several formulas , including specific identification , first - in first - out ( FIFO ) , or average cost . Costs include all costs of purchase , costs of conversion and other costs that are incurred in bringing the inventories to their present location and condition . Costs of goods made by the businesses include material , labor , and allocated overhead . The costs of those goods which are not yet sold are deferred as costs of inventory until the inventory is sold or written down in value .
  Fact 0.0: Costs are associated with particular goods using one of several formulas.
  Fact 0.1: The formulas include specific identification.
  Fact 0.2: The formulas include first-in first-out (FIFO).
  Fact 0.3: The formulas include average cost.
  Fact 0.4: Costs include all costs of purchase.
  Fact 0.5: Costs include all costs of conversion.
  Fact 0.6: Costs include other costs that are incurred in bringing the inventories to their present location and condition.
  Fact 0.7: Costs of goods made by businesses include material.
  Fact 0.8: Costs of goods made by businesses include labor.
  Fact 0.9: Costs of goods made by businesses include allocated overhead.
  Fact 0.10: Costs of goods which are not yet sold are deferred as costs of inventory.
  Fact 0.11: Costs of goods which are not yet sold are deferred until the inventory is sold.
  Fact 0.12: Costs of goods which are not yet sold are deferred until the inventory is written down in value.

Retrieved items:
<item 1>
Cost of goods sold
The average cost method relies on average unit cost to calculate cost of units sold and ending inventory . Several variations on the calculation may be used ,
including weighted average and moving average . First - In First - Out ( FIFO ) assumes that the items purchased or produced first are sold first . Costs of
inventory per unit or item are determined at the time made or acquired . The oldest cost ( i.e. , the first in ) is then matched against revenue and assigned to
cost of goods sold . Last - In First - Out ( LIFO ) is the reverse of FIFO . Some systems permit determining the costs of goods at the time acquired or made ,
but assigning costs to goods sold under the assumption that the goods made or acquired last are sold first . Costs of specific goods acquired or made are added
to a pool of costs for the type of goods . Under this system , the business may maintain costs under FIFO but track an offset in the form of a LIFO reserve .
Such reserve ( an asset or contra - asset ) represents the difference in cost of inventory under the FIFO and LIFO assumptions . Such amount may be different
for financial reporting and tax purposes in the United States .
</item 1>

<item 2>
Cost of goods sold
Most businesses make more than one of a particular item . Thus , costs are incurred for multiple items rather than a particular item sold . Determining how much
of each of these components to allocate to particular goods requires either tracking the particular costs or making some allocations of costs . Parts and raw
materials are often tracked to particular sets ( e.g. , batches or production runs ) of goods , then allocated to each item . Labor costs include direct labor
and indirect labor . Direct labor costs are the wages paid to those employees who spend all their time working directly on the product being manufactured .
Indirect labor costs are the wages paid to other factory employees involved in production . Costs of payroll taxes and fringe benefits are generally included in
labor costs , but may be treated as overhead costs . Labor costs may be allocated to an item or set of items based on timekeeping records . Costs of materials
include direct raw materials , as well as supplies and indirect materials . Where non-incidental amounts of supplies are maintained , the taxpayer must keep
inventories of the supplies for income tax purposes , charging them to expense or cost of goods sold as used rather than as purchased .
</item 2>

<item 3>
Cost of goods sold
Cost of goods sold - wikipedia Cost of goods sold Jump to : navigation , search Accounting Historical cost Constant purchasing power Management Tax Major types
( show ) Auditing Cost Forensic Financial Fund Governmental Management Social Tax Key concepts ( show ) Accounting period Accrual Constant purchasing power
Economic entity Fair value Going concern Historical cost Matching principle Materiality Revenue recognition Unit of account Selected accounts ( show ) Assets
Cash Cost of goods sold Depreciation / Amortization Equity Expenses Goodwill Liabilities Profit Revenue Accounting standards ( show ) Generally - accepted
principles Generally - accepted auditing standards Convergence International Financial Reporting Standards International Standards on Auditing Management
Accounting Principles Financial statements ( show ) Annual report Balance sheet Cash - flow Equity Income Management discussion Notes to the financial
statements Bookkeeping ( show ) Bank reconciliation Debits and credits Double - entry system FIFO and LIFO Journal Ledger / General ledger T accounts Trial
balance Auditing ( show ) Financial Internal Firms Report People and organizations ( show ) Accountants Accounting organizations Luca Pacioli Development ( show
) History Research Positive accounting Sarbanes -- Oxley Act Business portal Cost of goods sold ( COGS ) refers to the carrying value of goods sold during a
particular period .
</item 3>

<item 4>
Cost of goods sold
Determining costs requires keeping records of goods or materials purchased and any discounts on such purchase . In addition , if the goods are modified , the
business must determine the costs incurred in modifying the goods . Such modification costs include labor , supplies or additional material , supervision ,
quality control and use of equipment . Principles for determining costs may be easily stated , but application in practice is often difficult due to a variety
of considerations in the allocation of costs . Cost of goods sold may also reflect adjustments . Among the potential adjustments are decline in value of the
goods ( i.e. , lower market value than cost ) , obsolescence , damage , etc . When multiple goods are bought or made , it may be necessary to identify which
costs relate to which particular goods sold . This may be done using an identification convention , such as specific identification of the goods , first - in -
first - out ( FIFO ) , or average cost . Alternative systems may be used in some countries , such as last - in - first - out ( LIFO ) , gross profit method ,
retail method , or combinations of these .
</item 4>

<item 5>
Cost of goods sold
Jump up ^ 26 USC 263A , 26 CFR 1.263 A-3 ( c ) Archived 2011 - 06 - 12 at the Wayback Machine ... Jump up ^ Presti & Naegele Newsletter FAQ : What is `` cost of
goods sold '' ? Jump up ^ IAS 2 , paragraph 11 . Jump up ^ ASC 330 - 10 - 30 - 1 ; IAS 2 , paragraphs 12 - 14 ; 26 USC 263A ( a ) ( 2 ) ... Jump up ^ ASC 330 -
10 - 30 - 10 . Jump up ^ ASC 330 - 10 - 30 - 9 . Jump up ^ IAS 2 , paragraph 24 . Jump up ^ ASC 330 - 10 - 35 ; IAS 2 , paragraphs 28 - 33 . Further reading (
edit ) Fox , Stephen C , Income Tax in the USA chapter 23 , 2013 edition ISBN 978 - 0 - 9851 - 8233 - 5 , ASIN B00BCSNOGG .
</item 5>

<item 6>
Cost of goods sold
Cost of goods sold may be the same or different for accounting and tax purposes , depending on the rules of the particular jurisdiction . Certain expenses are
included in COGS . Expenses that are included in COGS can not be deducted again as a business expense . COGS expenses include : The cost of products or raw
materials , including freight or shipping charges ; The cost of storing products the business sells ; Direct labor costs for workers who produce the products ;
and Factory overhead expenses . Depreciation Importance of inventories ( edit ) Inventories have a significant effect on profits . A business that makes or buys
goods to sell must keep track of inventories of goods under all accounting and income tax rules . An example illustrates why . Fred buys auto parts and resells
them . In 2008 , Fred buys $100 worth of parts . He sells parts for $80 that he bought for $30 , and has $70 worth of parts left . In 2009 , he sells the
remainder of the parts for $180 . If he keeps track of inventory , his profit in 2008 is $50 , and his profit in 2009 is $110 , or $160 in total .
</item 6>

<item 7>
Cost of goods sold
U.S. Internal Revenue Service Publication 334 , Tax Guide for Small Business , pages 27 -- 29 . U.S. Financial Accounting Standards Board ASC 330 . ( hide )
Accounting Type Financial accounting Cost accounting Statements Balance sheet Income statement Cash flow statement Terms Revenue Cost of goods sold Operating
expenses Capital expenditure Depreciation Gross profit Net profit Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Cost_of_goods_sold&oldid=834402233 '' Categories : Inventory Accounting terminology Costs Sales Hidden categories :
All articles with dead external links Articles with dead external links from April 2018 Webarchive template wayback links Talk Contents About Wikipedia Deutsch
فارسی Bahasa Indonesia עברית Lietuvių 日本 語 Polski Русский Українська 中文 3 more Edit links This page was last edited on 5 April 2018 , at 14 : 57 . Text is
available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and
Privacy Policy .
</item 7>

<item 8>
Cost of goods sold
Costs are associated with particular goods using one of the several formulas , including specific identification , first - in first - out ( FIFO ) , or average
cost . Costs include all costs of purchase , costs of conversion and other costs that are incurred in bringing the inventories to their present location and
condition . Costs of goods made by the businesses include material , labor , and allocated overhead . The costs of those goods which are not yet sold are
deferred as costs of inventory until the inventory is sold or written down in value . Contents ( hide ) 1 Overview 2 Importance of inventories 3 Cost of goods
for resale 4 Cost of goods made by the business 5 Identification conventions 6 Example 7 Write - downs and allowances 8 Alternative views 9 Other terms 10 See
also 11 References 12 Further reading 12.1 Formal guidance Overview ( edit ) Many businesses sell goods that they have bought or produced . When the goods are
bought or produced , the costs associated with such goods are capitalized as part of inventory ( or stock ) of goods . These costs are treated as an expense in
the period the business recognizes income from sale of the goods .
</item 8>

<item 9>
Cost of goods sold
Such amount may be different for financial reporting and tax purposes in the United States . Dollar Value LIFO . Under this variation of LIFO , increases or
decreases in the LIFO reserve are determined based on dollar values rather than quantities . Retail inventory method . Resellers of goods may use this method to
simplify recordkeeping . The calculated cost of goods on hand at the end of a period is the ratio of cost of goods acquired to the retail value of the goods
times the retail value of goods on hand . Cost of goods acquired includes beginning inventory as previously valued plus purchases . Cost of goods sold is then
beginning inventory plus purchases less the calculated cost of goods on hand at the end of the period . Example ( edit ) Jane owns a business that resells
machines . At the start of 2009 , she has no machines or parts on hand . She buys machines A and B for 10 each , and later buys machines C and D for 12 each .
All the machines are the same , but they have serial numbers . Jane sells machines A and C for 20 each . Her cost of goods sold depends on her inventory method
.
</item 9>

<item 10>
Cost of goods sold
Other terms ( edit ) Net sales = gross sales -- ( customer discounts , returns , and allowances ) Gross profit = net sales -- cost of goods sold Operating
profit = gross profit -- total operating expenses Net profit = operating profit -- taxes -- interest Net profit = net sales -- cost of goods sold -- operating
expense -- taxes -- interest See also ( edit ) Cost of Revenue Inventory Average cost method List of business and finance abbreviations Accounting standards
Income tax in the USA References ( edit ) Jump up ^ U.S. Financial Accounting Standards Board ( FASB ) Accounting Standards Codification ASC 330 - 10 - 30 - 1 ;
International Accounting Standards IAS 2 , paragraph 10 . Jump up ^ IAS 2 , paragraph 34 . Jump up ^ This includes manufacturing from parts or raw materials .
Jump up ^ ASC 330 - 10 - 30 - 2 . Jump up ^ ASC 330 - 10 - 30 - 1 ; IAS 2 , paragraph 11 .
</item 10>
=== END ITEM id=ca2e7541a8e0f0cc ===

=== ITEM id=a65868fb20aeb95e ===
Gold sentences and facts:
Sentence 0: The music video for `` Total Eclipse of the Heart '' was directed by Russell Mulcahy and was filmed on location at the Holloway Sanatorium , a large Victorian Gothic hospital near Virginia Water , Surrey , England . The video features Bonnie Tyler clad in white , dreaming or fantasizing about her students in a boys ' boarding school . Young men are seen dancing and participating in various school activities and singing in a choir .
  Fact 0.0: The music video for 'Total Eclipse of the Heart' was directed by Russell Mulcahy.
  Fact 0.1: The music video for 'Total Eclipse of the Heart' was filmed on location at the Holloway Sanatorium.
  Fact 0.2: The Holloway Sanatorium is a large Victorian Gothic hospital.
  Fact 0.3: The Holloway Sanatorium is near Virginia Water, Surrey, England.
  Fact 0.4: The music video features Bonnie Tyler.
  Fact 0.5: In the music video, Bonnie Tyler is clad in white.
  Fact 0.6: In the music video, Bonnie Tyler is dreaming or fantasizing about her students in a boys' boarding school.
  Fact 0.7: In the music video, young men are seen dancing.
  Fact 0.8: In the music video, young men are seen participating in various school activities.
  Fact 0.9: In the music video, young men are seen singing in a choir.

Retrieved items:
<item 1>
Total Eclipse of the Heart
Retrieved 20 August 2014 . Jump up ^ DeGagne , Mike . `` Bonnie Tyler -- Total Eclipse of the Heart -- Song Review '' . AllMusic . All Media Network . Retrieved
12 January 2014 . Jump up ^ Guarisco , Donald A. `` Bonnie Tyler -- Faster Than the Speed of Night -- Album Review '' . AllMusic . All Media Network . Retrieved
13 January 2014 . Jump up ^ Brandle , Lars ( 8 March 2013 ) . `` Britain Counts On Bonnie Tyler to Play Eurovision Hero '' . Billboard . Retrieved 19 August
2014 . Jump up ^ Frame , Pete ( 1999 ) . Pete Frame 's rockin ' around Britain : rock'n'roll landmarks of the UK and Ireland . London : Omnibus Press . p. 174 .
ISBN 978 - 0 - 71196 - 973 - 5 . Jump up ^ `` Chelsea great Gianfranco Zola : ' I do n't star in Bonnie Tyler 's Total Eclipse of the Heart video ' '' . Off the
Post .
</item 1>

<item 2>
Total Eclipse of the Heart
Total Eclipse of the Heart - wikipedia Total Eclipse of the Heart Jump to : navigation , search For the episode of `` The Vampire Diaries '' , see Total Eclipse
of the Heart ( The Vampire Diaries ) . `` Total Eclipse of the Heart '' Artwork for UK and some European releases Single by Bonnie Tyler from the album Faster
Than the Speed of Night B - side `` Take Me Back '' ( UK ) `` Straight from the Heart '' ( US ) Released Format 7 '' 12 '' CD Length 6 : 59 ( album version ) 5
: 53 ( 2013 re-recording for the album Rocks and Honey ) 5 : 32 ( music video version ) 4 : 35 ( 2011 single version ) 4 : 30 ( single version ) 3 : 49 ( 2005
re-recording for the album Wings ) Label Columbia Songwriter ( s ) Jim Steinman Producer ( s ) Jim Steinman Bonnie Tyler singles chronology `` Sayonara Tokyo ''
( 1981 ) `` Total Eclipse of the Heart '' ( 1983 ) `` Take Me Back '' ( 1983 ) `` Sayonara Tokyo '' ( 1981 ) `` Total Eclipse of the Heart
</item 2>

<item 3>
Total Eclipse of the Heart
He also hand - picked the recording band for the song . `` Total Eclipse of the Heart '' had to be shortened for radio play . Tyler did not believe that the
song was radio - friendly at its full length ; the song was reduced from seven minutes and two seconds to four minutes and thirty seconds . The power ballad
remains Tyler 's most successful song , peaking at No. 1 in the United States , South Africa , Australia , New Zealand , Canada and the United Kingdom . At its
peak , it sold 60,000 copies per day , and approximately 6 million copies in total . It won the Variety Club award in the UK for best single of 1983 . The song
also made number 82 of VH1 's top 100 love songs . `` Total Eclipse of the Heart '' A sample of `` Total Eclipse of the Heart '' , performed by Bonnie Tyler .
Problems playing this file ? See media help . Tyler told Record Mirror that she thought the song was about `` someone who wants to love so badly she 's lying
there in complete darkness . ''
</item 3>

<item 4>
Total Eclipse of the Heart
Young men are seen dancing and participating in various school activities and singing in a choir . A long - running urban legend is that the boy who appears
throughout the video and who shakes Tyler 's hand at the end is former Italian footballer Gianfranco Zola . In a 2012 interview , Zola confirmed that he did not
appear in the video . Live performances ( edit ) Since the song 's release , Tyler performs `` Total Eclipse of the Heart '' in every concert she performs . ``
I sing it much better now than I used to , '' she told The Huffington Post . `` I think my voice is probably not as husky as it was , I think it 's mellowed a
bit . '' The song was performed at the 26th Annual Grammy Awards , held at the Shrine Auditorium , Los Angeles , on 28 February 1984 . Tyler performed a brief ,
live performance of this song on a Royal Caribbean cruise during the solar eclipse of 21 August 2017 . Live recordings of Tyler performing the song have been
released on her live albums Bonnie Tyler Live ( 2006 ) and Live in Germany 1993 ( 2011 ) .
</item 4>

<item 5>
Total Eclipse of the Heart
) `` Sayonara Tokyo '' ( 1981 ) `` Total Eclipse of the Heart '' ( 1983 ) `` Take Me Back '' ( 1983 ) Music video `` Total Eclipse Of The Heart '' on YouTube ``
Total Eclipse of the Heart '' is a song recorded by Welsh singer Bonnie Tyler . It was written and produced by Jim Steinman , and released on Tyler 's fifth
studio album , Faster Than the Speed of Night ( 1983 ) . The song was released as a single by Columbia Records on 11 February 1983 in the United Kingdom and on
31 May 1983 in the United States . The song became Tyler 's biggest career hit , topping the UK Singles Chart , and becoming the fifth - best - selling single
in 1983 in the United Kingdom . In the United States , the single spent four weeks at the top of the charts , and was Billboard 's number - six song of the year
for 1983 .
</item 5>

<item 6>
Total Eclipse of the Heart
Jump up ^ Wilson , Jess ( 20 March 2015 ) . `` Solar Eclipse : Spotify reveals streaming of Bonnie Tyler 's Total Eclipse of the Heart is up 214 per cent '' .
Daily Mirror . Retrieved 2 April 2015 . Jump up ^ Caulfield , Keith ( 21 August 2017 ) . `` ' Total Eclipse of the Heart ' Sales Rise More Than 500 % '' .
Billboard . Retrieved 21 August 2017 . Jump up ^ Pelletiere , Nicole ( 21 August 2017 ) . `` ' Total Eclipse of the Heart ' hits no . 1 on iTunes for solar
eclipse '' . ABC News . Retrieved 22 August 2017 . Jump up ^ Bruner , Raisa ( 16 August 2017 ) . `` Bonnie Tyler Will Sing ' Total Eclipse of the Heart ' During
the Actual Eclipse '' . Time . Retrieved 17 August 2017 . Jump up ^ Balfe , John ( 12 August 2014 ) . `` 80s Power Ballad queen Bonnie Tyler announced for
Electric Picnic '' . entertainment.ie .
</item 6>

<item 7>
Total Eclipse of the Heart
230 . Jump up ^ Scott , David ( 16 April 2011 ) . `` Total Eclipse of the Heart : Literal Video Version '' . Retrieved 26 August 2017 . Jump up ^ Benedictus ,
Leo ( 21 April 2010 ) . `` A word on our sponsors : Total eclipse of the card '' . The Guardian . Retrieved 20 August 2014 . Jump up ^ `` Westpac brings in
Bonnie Tyler to sing at couple 's fantasy wedding '' . mUmBRELLA. 21 November 2012 . Retrieved 22 August 2014 . Jump up ^ Matthewson , Samantha ( 1 August 2017
) . `` This ' Total Eclipse of the Heart ' Parody Video for the 2017 Solar Eclipse Is Just Awesome '' . Space.com . Retrieved 3 August 2017 .
</item 7>

<item 8>
Total Eclipse of the Heart
Donald A. Guarisco , also from AllMusic , retrospectively reviewed Faster Than the Speed of Night , and noted the song as an `` epic ballad , '' describing the
whole album as `` rock at its most melodramatic . '' Jim Beviglia from American Songwriter said that Tyler 's raspy vocals helped to legitimize the `` melodrama
inherent in the lyrics , '' and described the song as a `` garment - rending , chest - beating ( and ) emotionally exhausting ballad '' that suits the throes of
a turbulent relationship . Music video ( edit ) Holloway Sanatorium steps , which feature in the video The music video for `` Total Eclipse of the Heart '' was
directed by Russell Mulcahy and was filmed on location at the Holloway Sanatorium , a large Victorian Gothic hospital near Virginia Water , Surrey , England .
The video features Bonnie Tyler clad in white , dreaming or fantasizing about her students in a boys ' boarding school . Young men are seen dancing and
participating in various school activities and singing in a choir .
</item 8>

<item 9>
Total Eclipse of the Heart
Off the Post . 16 May 2012 . Retrieved 5 September 2014 . Jump up ^ `` Bonnie Tyler : ' Forget being a star - do it for the love of it ' '' . The Guardian . 10
May 2009 . Retrieved 16 March 2014 . Jump up ^ Frost , Caroline ( 7 May 2013 ) . `` Bonnie Tyler Interview : ' I Wo n't Mind If Believe In Me Gets Nil Points At
Eurovision Song Contest ' '' . The Huffington Post . Retrieved 26 August 2014 . Jump up ^ `` Jackson , Police top Grammy Awards list '' . The Michigan Daily .
29 February 1984 . Retrieved 26 August 2014 . Jump up ^ Mahtani , Melissa ( 21 August 2017 ) . `` Bonnie Tyler sings ' Total Eclipse ' live on CNN '' ( Press
release ) . CNN . Retrieved 22 August 2017 . Jump up ^ Reed , Ryan ( 16 August 2017 ) . `` Bonnie Tyler Will Sing ' Total Eclipse of the Heart ' During Solar
Eclipse '' . Rolling Stone .
</item 9>

<item 10>
Total Eclipse of the Heart
Retrieved 19 August 2014 . ^ Jump up to : Beviglia , Jim ( 24 February 2014 ) . `` Bonnie Tyler , `` Total Eclipse Of The Heart '' `` . American Songwriter .
Retrieved 25 February 2014 . Jump up ^ Clark , Dick ( 20 September 1983 ) . `` Bonnie Tyler aims for ' total eclipse ' of charts '' . Times - News . Retrieved
25 August 2014 . Jump up ^ Adams , Cameron ( 26 October 2006 ) . `` Meat Loaf 's a Hell raiser '' . Herald Sun . Jump up ^ Champion , Edward ( 12 September 2008
) . `` The Bat Segundo Show : Bonnie Tyler '' . The Bat Segundo Show . Retrieved 8 May 2013 . Jump up ^ Freyne , Patrick ( 20 August 2014 ) . `` Bonnie Tyler :
' There 's nothing I wo n't talk about ' '' . The Irish Times . Retrieved 20 August 2014 . Jump up ^ DeGagne , Mike .
</item 10>
=== END ITEM id=a65868fb20aeb95e ===

=== ITEM id=f736fcf664433dbb ===
Gold sentences and facts:
Sentence 0: On 19 February 2007 Disney lost a court case in Los Angeles which ruled their `` misguided claims '' to dispute the licensing agreements with Slesinger , Inc. were unjustified , but a federal ruling of 28 September 2009 , again from Judge Florence - Marie Cooper , determined that the Slesinger family had granted all trademark and copyright rights to Disney , although Disney must pay royalties for all future use of the characters . Both parties have expressed satisfaction with the outcome .
  Fact 0.0: On 19 February 2007, Disney lost a court case in Los Angeles.
  Fact 0.1: That court ruled that Disney's misguided claims to dispute the licensing agreements with Slesinger, Inc. were unjustified.
  Fact 0.2: A federal ruling of 28 September 2009 was again from Judge Florence-Marie Cooper.
  Fact 0.3: The federal ruling of 28 September 2009 determined that the Slesinger family had granted all trademark and copyright rights to Disney.
  Fact 0.4: Disney must pay royalties for all future use of the characters.
  Fact 0.5: Both parties have expressed satisfaction with the outcome.

Retrieved items:
<item 1>
Winnie-the-Pooh
were unjustified , but a federal ruling of 28 September 2009 , again from Judge Florence - Marie Cooper , determined that the Slesinger family had granted all
trademark and copyright rights to Disney , although Disney must pay royalties for all future use of the characters . Both parties have expressed satisfaction
with the outcome . Adaptations Theatre Winnie - the - Pooh at the Guild Theater Sue Hastings Marionettes , 1931 Winnie - the - Pooh , a play in three acts
Dramatized by Kristin Sergel , Dramatic Publishing Company , 1957 Winnie - the - Pooh , a musical comedy in two acts Lyrics by A.A. Milne and Kristin Sergel ,
Music by Allan Jay Friedman , Book by Kristin Sergel , Dramatic Publishing Company , 1964 A Winnie - the - Pooh Christmas Tail , In Which Winnie - the - Pooh
and His Friends Help Eeyore Have a Very Merry Christmas ( Or a Very Happy Birthday . ) Book , Music and Lyrics by James W. Rogers , Dramatic Publishing Company
, 1977 `` Bother !
</item 1>

<item 2>
Winnie-the-Pooh
Parker Brothers introduced A.A. Milne 's Winnie - the - Pooh Game in 1933 , again with Pooh in his red shirt . In the 1940s , Agnes Brush created the first
plush dolls with Pooh in his red shirt . Shepard had drawn Pooh with a shirt as early as the first Winnie - The - Pooh book , which was subsequently coloured
red in later coloured editions . Disney ownership era ( 1966 -- present ) Main article : Winnie the Pooh ( franchise ) After Slesinger 's death in 1953 , his
wife , Shirley Slesinger Lasswell , continued developing the character herself . In 1961 , she licensed rights to Walt Disney Productions in exchange for
royalties in the first of two agreements between Stephen Slesinger , Inc. and Disney . The same year , A.A. Milne 's widow , Daphne Milne , also licensed
certain rights , including motion picture rights , to Disney . Since 1966 , Disney has released numerous animated productions starring Winnie the Pooh and
related characters .
</item 2>

<item 3>
Winnie-the-Pooh
Benjamin Hoff used Milne 's characters in The Tao of Pooh and The Te of Piglet to explain Taoism . Similarly , Frederick Crews wrote essays about the Pooh books
in abstruse academic jargon in The Pooh Perplex and Postmodern Pooh to satirise a range of philosophical approaches . Pooh and the Philosophers by John T.
Williams uses Winnie the Pooh as a backdrop to illustrate the works of philosophers including Descartes , Kant , Plato and Nietzsche . One of the best known
characters in British children 's literature , a 2011 poll saw Winnie the Pooh voted onto the list of icons of England . Forbes magazine ranked Pooh the most
valuable fictional character in 2002 , with merchandising products alone generating more than $5.9 billion that year . In 2005 , Pooh generated $6 billion , a
figure surpassed by only Mickey Mouse . In 2006 , Pooh received a star on the Hollywood Walk of Fame , marking the 80th birthday of Milne 's creation .
</item 3>

<item 4>
Winnie-the-Pooh
OCLC WorldCat . Retrieved 15 January 2016 . Jump up ^ `` Tigger comes to the forest : and other stories '' . OCLC WorldCat . Retrieved 15 January 2016 . Jump up
^ `` His Master 's Voice Speaks Again '' . Playthings . November 1932 . Jump up ^ Fleming Jr. , Mike . `` Disney Sets Live - Action ' Winnie The Pooh ' Film ;
Alex Ross Perry To Write '' . Deadline . Retrieved 6 April 2015 . Jump up ^ Lincoln , Ross A. ( 18 November 2016 ) . `` Marc Forster To Helm Live - Action '
Christopher Robin ' Based On ' Winnie The Pooh ' Character '' . Deadline . Jump up ^ `` Russian animation in letters and figures `` Winnie the Pooh '' `` .
Animator.ru . Retrieved 9 March 2015 . Jump up ^ Biography : Willie Rushton . BBC . Retrieved 11 March 2015 . Jump up ^ `` Mini Adventures of Winnie the Pooh ''
. IMDb.
</item 4>

<item 5>
Winnie-the-Pooh
The bear cub was purchased from a hunter for $20 by Canadian Lieutenant Harry Colebourn in White River , Ontario , Canada , while en route to England during the
First World War . He named the bear `` Winnie '' after his adopted hometown in Winnipeg , Manitoba . `` Winnie '' was surreptitiously brought to England with
her owner , and gained unofficial recognition as The Fort Garry Horse regimental mascot . Colebourn left Winnie at the London Zoo while he and his unit were in
France ; after the war she was officially donated to the zoo , as she had become a much loved attraction there . Pooh the swan appears as a character in its own
right in When We Were Very Young . In the first chapter of Winnie - the - Pooh , Milne offers this explanation of why Winnie - the - Pooh is often called simply
`` Pooh '' : But his arms were so stiff ... they stayed up straight in the air for more than a week , and whenever a fly came and settled on his nose he had to
blow it off . And I think -- but I am not sure -- that that is why he is always called Pooh .
</item 5>

<item 6>
Winnie-the-Pooh
Jump up ^ `` The Curse of Pooh . '' Fortune . Jump up ^ `` The Pooh Files '' The Albion Monitor . Jump up ^ Nelson , Valerie J ( 20 July 2007 ) . `` Shirley
Slesinger Lasswell , 84 ; fought Disney over Pooh royalties '' . Los Angeles Times . Retrieved 14 August 2007 . Jump up ^ `` Judge dismisses Winnie the Pooh
lawsuit '' The Disney Corner . Jump up ^ James , Meg ( 26 September 2007 ) . `` Disney wins lawsuit ruling on Pooh rights '' . Los Angeles Times . Retrieved 26
September 2007 . Jump up ^ `` Winnie the Pooh goes to court '' USA Today Jump up ^ `` USATODAY.com -- Justices wo n't hear copyright appeal by relative of
Winnie the Pooh '' . USA Today . Associated Press . Jump up ^ `` Disney loses court battle in Winnie the Pooh copyright case '' . ABC News . 17 February 2007 .
Retrieved 15 May 2008 .
</item 6>

<item 7>
Winnie-the-Pooh
Though the Disney corporation was sanctioned by a judge for destroying forty boxes of evidential documents , the suit was later terminated by another judge when
it was discovered that Slesinger 's investigator had rummaged through Disney 's garbage to retrieve the discarded evidence . Slesinger appealed the termination
, and on 26 September 2007 , a three - judge panel upheld the lawsuit dismissal . After the Copyright Term extension Act of 1998 , Clare Milne , Christopher
Milne 's daughter , attempted to terminate any future US copyrights for Stephen Slesinger , Inc . After a series of legal hearings , Judge Florence - Marie
Cooper of the US District Court in California found in favour of Stephen Slesinger , Inc. , as did the United States Court of Appeals for the Ninth Circuit . On
26 June 2006 , the US Supreme Court refused to hear the case , sustaining the ruling and ensuring the defeat of the suit . On 19 February 2007 Disney lost a
court case in Los Angeles which ruled their `` misguided claims '' to dispute the licensing agreements with Slesinger , Inc.
</item 7>

<item 8>
Winnie-the-Pooh
17 February 2007 . Retrieved 15 May 2008 . Jump up ^ James , Meg ( 29 September 2009 ) . `` Pooh rights belong to Disney , judge rules '' . Los Angeles Times .
Retrieved 5 October 2009 . Jump up ^ Shea , Joe ( 4 October 2009 ) . `` The gordian knot of Pooh rights is finally untied in federal court '' . The American
Reporter . Retrieved 5 October 2009 . Jump up ^ `` Hastings Marionettes : Will Open Holiday Season at Guild Theatre on Saturday '' . The New York Times . 22
December 1931 . p. 28 . Jump up ^ `` Ian Carmichael And Full Cast -- The House at Pooh Corner -- HMV Junior Record Club -- UK -- 7EG 117 '' . 45cat. 23 July
2010 . Retrieved 5 November 2011 . Jump up ^ `` Winnie the Pooh '' . OCLC WorldCat . Retrieved 15 January 2016 .
</item 8>

<item 9>
Winnie-the-Pooh
A further special story , Winnie - the - Pooh Meets the Queen , was published in 2016 to mark the 90th anniversary of Milne 's creation and the 90th birthday of
Elizabeth II . It sees Winnie the Pooh meet the Queen at Buckingham Palace . Stephen Slesinger On 6 January 1930 , Stephen Slesinger purchased US and Canadian
merchandising , television , recording and other trade rights to the `` Winnie - the - Pooh '' works from Milne for a $1000 advance and 66 % of Slesinger 's
income , creating the modern licensing industry . By November 1931 , Pooh was a $50 million - a-year business . Slesinger marketed Pooh and his friends for more
than 30 years , creating the first Pooh doll , record , board game , puzzle , US radio broadcast ( NBC ) , animation , and motion picture film . Red shirt Pooh
The first time Pooh and his friends appeared in colour was 1932 , when he was drawn by Slesinger in his now - familiar red shirt and featured on an RCA Victor
picture record .
</item 9>

<item 10>
Winnie-the-Pooh
These have included theatrical featurettes , television series , and direct - to - video films , as well as the theatrical feature - length films The Tigger
Movie , Piglet 's Big Movie , Pooh 's Heffalump Movie , and Winnie the Pooh . Merchandising revenue dispute Pooh videos , soft toys , and other merchandise
generate substantial annual revenues for Disney . The size of Pooh stuffed toys ranges from Beanie and miniature to human - sized . In addition to the stylised
Disney Pooh , Disney markets Classic Pooh merchandise which more closely resembles E.H. Shepard 's illustrations . In 1991 , Stephen Slesinger , Inc. filed a
lawsuit against Disney which alleged that Disney had breached their 1983 agreement by again failing to accurately report revenue from Winnie the Pooh sales .
Under this agreement , Disney was to retain approximately 98 % of gross worldwide revenues while the remaining 2 % was to be paid to Slesinger . In addition ,
the suit alleged that Disney had failed to pay required royalties on all commercial exploitation of the product name .
</item 10>
=== END ITEM id=f736fcf664433dbb ===
